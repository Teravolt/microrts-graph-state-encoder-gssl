"""
Train graph state encoder using DINO
Paper: https://arxiv.org/pdf/2104.14294
"""

import argparse
from argparse import Namespace

from pathlib import Path
from copy import deepcopy

import numpy as np

import torch
from torch import optim
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_edge
from torch_geometric.transforms.normalize_features import NormalizeFeatures

from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from accelerate.utils import set_seed

from diffusers import get_cosine_schedule_with_warmup

import wandb

from replay_parser.parser import parse_replay_dataset

from nn.state_encoder import GNNStateEncoderV2BGRL

def prepare_dataloader(config: Namespace):
    """
    Prepare dataloader

    :param config: Script config
    :returns Dataloader for state dataset, node dimensions,
             and edge dimensions
    """

    replay_data = parse_replay_dataset(config)

    node_dims = -1
    edge_dims = -1

    dataset = []
    for filename, pid, trace in replay_data:
        _, players, *_ = filename.split('.')
        player_1, _, player_2, *_ = players.split('-')
        players = [player_1, player_2]

        if players[pid] in config.ignore_players:
            # Don't add player data to dataset
            continue

        for state, _ in trace:

            if node_dims == -1:
                node_dims = state.x.shape[-1]

            if edge_dims == -1:
                edge_dims = state.edge_attr.shape[-1]

            dataset.append(state)

    loader = DataLoader(dataset, batch_size=config.batch_size,
                        generator=config.rng, shuffle=True)
    return loader, node_dims, edge_dims

def create_model(node_dims: int, edge_dims: int, hidden_dims: int,
                 _config: Namespace):
    """
    Create model

    :param node_dims: Number of nodes features
    :param edge_dims: Number of edge features
    :param hidden_dims: Hidden dimensions
    :param config: Script config
    :returns: GNN state encoder model
    """

    state_enc = GNNStateEncoderV2BGRL(node_dims, edge_dims, hidden_dims)

    return state_enc

def compute_loss(proj_embedding_s1: torch.Tensor, proj_embedding_s2: torch.Tensor,
                 node_embedding_t1: torch.Tensor, node_embedding_t2: torch.Tensor):
    """
    Compute BGRL loss

    :param proj_embedding_s1: Projection from student for graph 1
    :param proj_embedding_s2: Projection from student for graph 2
    :param node_embedding_t1: Projection from teacher for graph 1
    :param node_embedding_t2: Projection from teacher for graph 2
    :returns: BGRL loss between the two node embeddings
    """

    loss_1 = F.cosine_similarity(proj_embedding_s1, node_embedding_t2.detach(), dim=-1).mean()
    loss_2 = F.cosine_similarity(proj_embedding_s2, node_embedding_t1.detach(), dim=-1).mean()
    loss = 2 - loss_1 - loss_2

    return loss

def perturb_state(state, config: Namespace):
    """
    Preturb state randomly

    :param state: State to perturb
    :param config: Script config
    :returns: Perturbed state
    """

    rand_val = torch.rand(1, generator=config.rng, device=config.device)

    # Perturbation 1: Scale
    scale_diff = config.max_scale_factor-config.min_scale_factor
    scale_factor = rand_val*(scale_diff)+(1.0-(scale_diff/2.0))
    state.edge_attr[:, 0] = scale_factor*state.edge_attr[:, 0]

    # # Perturbation 2: Rotation
    rotation_factor = 2*rand_val*np.pi
    state.edge_attr[:, 1] += rotation_factor
    state.edge_attr[:, 1] = torch.where(
        state.edge_attr[:, 1] > 2*np.pi,
        state.edge_attr[:, 1]-2*np.pi,
        state.edge_attr[:, 1])

    # Perturbation 3: Drop edges
    edge_index, edge_mask = dropout_edge(state.edge_index)
    state.edge_index = edge_index
    state.edge_attr = state.edge_attr[edge_mask, :]

    return state

def training_loop(config: Namespace, debug_mode=False):
    """
    Training loop

    :param config: Script config
    :param debug_mode: True if using debug mode
    """

    accelerator: Accelerator = None
    set_seed(config.seed)
    if not debug_mode:

        grad_accumulation_plugin = GradientAccumulationPlugin(
            num_steps=config.grad_accumulation_steps,
            adjust_scheduler=True,
            sync_with_dataloader=True)

        accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_plugin=grad_accumulation_plugin,
            cpu=(config.device == 'cpu'))

    dataloader, node_dims, edge_dims = prepare_dataloader(config)
    student_model = create_model(node_dims, edge_dims, config.hidden_dims, config)
    teacher_model = deepcopy(student_model)

    optimizer = optim.AdamW(student_model.parameters(),
                            lr=config.learning_rate)

# #     scheduler = CosineAnnealingLR(
# #         optimizer,
# #         T_max=config.num_train_epochs)
    # scheduler = optim.lr_scheduler.ExponentialLR(
    #     optimizer,
    #     config.lr_exp_schedule_gamma)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.lr_warmup_steps, len(dataloader)*config.num_train_epochs)

#     scheduler = CosineAnnealingWarmRestarts(
#         optimizer,
#         T_0=config.lr_warmup_steps)
        # last_epoch=config.num_train_epochs*len(train_dataloader))

    if accelerator:
        student_model, optimizer, dataloader, scheduler \
            = accelerator.prepare(student_model, optimizer, dataloader, scheduler)

    wandb_run = None
    if not debug_mode:
        wandb_run = wandb.init(project=config.project_name, entity=None,
                            job_type='training',
                            name=config.run_name,
                            config=config)

        wandb_run.define_metric("epoch")
        wandb_run.define_metric("training_step")

        wandb_run.define_metric('val_total_loss', step_metric='epoch')

        wandb_run.define_metric('step_loss', step_metric='training_step')

        wandb_run.define_metric('epoch_loss', step_metric='training_step')

        wandb_run.define_metric('lr', step_metric='training_step')

    num_steps = 0
    norm_feature_fn = NormalizeFeatures(attrs=['x'])
    for epoch in range(config.num_train_epochs):
        student_model.train()

        print(f"Epoch {epoch}")

        epoch_loss = 0
        num_iters = 0

        for _, batch in enumerate(dataloader):

            optimizer.zero_grad()

            norm_batch = norm_feature_fn(batch)

            state_1 = deepcopy(norm_batch)
            state_2 = deepcopy(norm_batch)

            # Perturb both states
            perturbed_state_1 = perturb_state(state_1, config)
            perturbed_state_2 = perturb_state(state_2, config)

            # perturbed_contrast_state = deepcopy(contrast_state)

            # rand_val = torch.rand(1, generator=config.rng, device=config.device)
            # scale_diff = config.max_scale_factor-config.min_scale_factor
            # scale_factor = rand_val*(scale_diff)+(1.0-(scale_diff/2.0))
            # perturbed_contrast_state.edge_attr[:, 0] = \
            #     scale_factor*perturbed_contrast_state.edge_attr[:, 0]

            _, _, proj_embedding_s1 = student_model(
                perturbed_state_1.x,
                perturbed_state_1.edge_index,
                perturbed_state_1.edge_attr,
                perturbed_state_1.batch)

            _, _, proj_embedding_s2 = student_model(
                perturbed_state_2.x,
                perturbed_state_2.edge_index,
                perturbed_state_2.edge_attr,
                perturbed_state_2.batch)

            with torch.no_grad():
                node_embedding_t1, _, _ = teacher_model(
                    perturbed_state_1.x,
                    perturbed_state_1.edge_index,
                    perturbed_state_1.edge_attr,
                    perturbed_state_1.batch)

                node_embedding_t2, _, _ = teacher_model(
                    perturbed_state_2.x,
                    perturbed_state_2.edge_index,
                    perturbed_state_2.edge_attr,
                    perturbed_state_2.batch)

            loss = compute_loss(
                proj_embedding_s1,
                proj_embedding_s2,
                node_embedding_t1,
                node_embedding_t2)

            # accelerator.print(f"Loss: {loss.item()}")
            if accelerator:
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(student_model.parameters(), 1.0)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)

            epoch_loss += loss.item()

            if wandb_run:
                wandb_run.log({'training_step': num_steps, 'step_loss': loss.item()})
                wandb_run.log({'training_step': num_steps, 'lr': scheduler.get_lr()[0]})
            else:
                print(f"Step loss: {loss.item()}")

            if num_steps % config.update_freq == 0:
                # ema_model.load_state_dict(model.state_dict())
                teacher_state_dict = teacher_model.state_dict()
                for key, parameters in student_model.state_dict().items():
                    teacher_state_dict[key] = config.ema_alpha*teacher_state_dict[key] \
                        + (1-config.ema_alpha)*parameters
                teacher_model.load_state_dict(teacher_state_dict)

            num_steps += 1
            num_iters += 1

            # Update the model parameters with the optimizer
            optimizer.step()
            scheduler.step()

        # Validate model
        # accelerator.print("Evaluating model")
        # eval_loop(epoch, model, val_dataloader, wandb_run)

        if wandb_run:
            wandb_run.log({'training_step': num_steps, 'epoch_loss': epoch_loss/num_iters})
        else:
            print(f"Epoch loss: {epoch_loss}")

    if config.save_model:
        # Save model to W&Bs

        torch.save(student_model.state_dict(), config.save_model)
        if wandb_run:
            model_name = config.save_model.stem
            model_art = wandb.Artifact(model_name, type='model')
            model_art.add_file(config.save_model)
            wandb_run.log_artifact(model_art)

    if wandb_run:
        wandb_run.finish()

def get_config():
    """
    Build config

    :returns: Config from command line arguments
    """

    parser = argparse.ArgumentParser(
        description='Train MicroRTS GNN State Encoder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Replay Config
    parser.add_argument('input_directory', help='Directory containing replays')
    parser.add_argument('--read_from_zip', action='store_true',
                        help='Read from zip file')
    parser.add_argument('--allow_coordinate', action='store_true',
                        help='Allow coordinates in states and actions')
    parser.add_argument('--unit_actions_to_ignore', nargs="*", type=str, default=[],
                        help='Unit actions to ignore')
    parser.add_argument('--state_representation', type=str, default='graph',
                        help="State representation to use")
    parser.add_argument('--max_replays', default=-1, type=int,
                        help='Maximum number of replays to read')
    parser.add_argument('--max_replay_length', default=-1, type=int,
                        help='Maximum replay length')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')
    parser.add_argument('--frame_skip_freq', default=5, type=int,
                        help='Number of frames to skip')
    parser.add_argument('--frame_number_start', default=0, type=int,
                        help='Offset to start replay parsing')

    # Training Config
    parser.add_argument("--batch_size", default=8, type=int,
                        help='Batch size')
    parser.add_argument("--hidden_dims", default=256, type=int,
                        help='Hidden dim size')
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", default=4e-7, type=float,
                        help='Optimizer learning rate')
    parser.add_argument("--lr_exp_schedule_gamma", default=0.99, type=float,
                        help="Gamma value for exponential lr scheduler")
    parser.add_argument("--lr_warmup_steps", default=1000, type=int,
                        help="Number of warmup steps for cosine scheduler")
    parser.add_argument("--update_freq", default=1, type=int,
                        help="Frequency to update siamese network")
    parser.add_argument("--ema_alpha", default=0.996, type=float,
                        help="Alpha value for Exponential Moving Average (EMA)")
    parser.add_argument('--grad_accumulation_steps', default=4, type=int,
                        help="Number of steps to accumulate gradients")
    parser.add_argument('--mixed_precision', default=None, type=str,
                        help="Mixed-precision training")
    parser.add_argument('--device', default=None, type=str,
                        help="Device to run model and training")
    parser.add_argument('--debug_mode', action='store_true',
                        help="Flag to turn on debugging mode.")

    # Model Config
    # parser.add_argument("--model_name", default="gnn-state-encoder", type=str,
    #                     help="Name of model")
    parser.add_argument("--save_model", default=None, type=Path,
                        help="Filename for model")

    # Weights and Biases
    parser.add_argument('--project_name', default="microrts-graph-state-ssl",
                        type=str, help="Name of project on W&Bs")
    parser.add_argument('--run_name', default="run-0",
                        type=str, help="Name of run on W&Bs")

    # Data Augmentation Config
    parser.add_argument("--min_scale_factor", default=0.5, type=float,
                        help="Minimum distance scaling factor")
    parser.add_argument("--max_scale_factor", default=1.5, type=float,
                        help="Maximum distance scaling factor")

    parser.add_argument("--student_temp", default=0.1, type=float,
                        help="Student temperature initial value")
    parser.add_argument("--teacher_temp", default=0.04, type=float,
                        help="Teacher temperature initial value")

    config = parser.parse_args()

    if config.device is None:
        config.device = torch.device(
            'cuda' if torch.cuda.is_available() \
                else 'mps' if torch.backends.mps.is_available() else 'cpu')

    config.alpha = [0.99, 0.99]
    config.theta = [0.99, 0.99]

    # DINO temperature parameters
    config.student_temp = torch.tensor(config.student_temp)
    config.teacher_temp = torch.tensor(config.teacher_temp)

    # config.frame_skip_freq = 5 # (avg. movement frames is 10)

    # Ignore RandomAI - Not really useful for training a model
    config.ignore_players = ['0']

    config.rng = torch.Generator(config.device).manual_seed(config.seed)

    return config

def main():
    """
    Main function
    """

    config = get_config()
    training_loop(config, debug_mode=config.debug_mode)

if __name__ == "__main__":
    main()
