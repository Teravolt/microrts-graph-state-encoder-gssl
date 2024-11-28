"""
Train graph state encoder
"""

import argparse
from argparse import Namespace

from copy import deepcopy

import torch
from torch import optim
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from accelerate.utils import set_seed

from diffusers import get_cosine_schedule_with_warmup

import wandb

from replay_parser.parser import parse_replay_dataset

from nn.state_encoder import GNNStateEncoder

def prepare_dataloader(config: Namespace):
    """
    Prepare dataloader

    :param config: Script config
    :returns Dataloader for state dataset, node dimensions, and edge dimensions
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
            # print(filename, players, players[pid])
            continue

        for state, _ in trace:

            contrast_state = deepcopy(state)
            if node_dims == -1:
                node_dims = state.x.shape[-1]

            if edge_dims == -1:
                edge_dims = state.edge_attr.shape[-1]

            dataset.append((state, contrast_state))

    return DataLoader(dataset, batch_size=config.batch_size, generator=config.rng, shuffle=True), \
        node_dims, edge_dims

def create_model(node_dims: int, edge_dims: int, hidden_dims: int, _config: Namespace):
    """
    Create model

    :param node_dims: Number of nodes features
    :param edge_dims: Number of edge features
    :param hidden_dims: Hidden dimensions
    :param config: Script config
    :returns: GNN state encoder model
    """

    state_enc = GNNStateEncoder(node_dims, edge_dims, hidden_dims)

    return state_enc

def compute_loss(graph_embedding_1: torch.Tensor, graph_embedding_2: torch.Tensor):
    """
    Compute SimCLR contrastive loss

    :param graph_embedding_1: Embedding from first graph
    :param graph_embedding_1: Embedding from second graph
    :returns: SimCLR loss between the two graph embeddings
    """

    batch_size = graph_embedding_1.shape[0]

    # Pairwise similarity
    norm_graph_embedding_1 = F.normalize(graph_embedding_1, p=2, dim=1)
    norm_graph_embedding_2 = F.normalize(graph_embedding_2, p=2, dim=1)

    graph_embeddings = torch.concat(
        [norm_graph_embedding_1, norm_graph_embedding_2], dim=0)

    sim = torch.matmul(graph_embeddings, graph_embeddings.T)
    sim = (1 - torch.eye(sim.shape[0]))*sim + 1e-8

    row_softmax = torch.log_softmax(sim, dim=0)
    col_softmax = torch.log_softmax(sim, dim=1)

    loss = 0
    for i in range(batch_size):
        loss += (row_softmax[i, batch_size+i] + col_softmax[batch_size+i, i])
    loss = loss / (2*batch_size)

    return -loss

def training_loop(config: Namespace, debug_mode=False):
    """
    Training loop

    :param config: Script config
    :param debug_mode: True if using debug mode
    """

    accelerator: Accelerator = None
    if not debug_mode:
        set_seed(config.seed)

        grad_accumulation_plugin = GradientAccumulationPlugin(
            num_steps=config.grad_accumulation_steps,
            adjust_scheduler=True,
            sync_with_dataloader=True)

        accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_plugin=grad_accumulation_plugin,
            cpu=(config.device == 'cpu'))

    dataloader, node_dims, edge_dims = prepare_dataloader(config)
    model = create_model(node_dims, edge_dims, config.hidden_dims, config)
    ema_model = deepcopy(model)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

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
        model, optimizer, dataloader, scheduler \
            = accelerator.prepare(model, optimizer, dataloader, scheduler)

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
    for epoch in range(config.num_train_epochs):
        model.train()

        print(f"Epoch {epoch}")

        epoch_loss = 0
        num_iters = 0

        for _, batch in enumerate(dataloader):

            optimizer.zero_grad()
            state, contrast_state = batch
            perturbed_contrast_state = deepcopy(contrast_state)

            rand_val = torch.rand(1, generator=config.rng, device=config.device)
            scale_diff = config.max_scale_factor-config.min_scale_factor
            scale_factor = rand_val*(scale_diff)+(1.0-(scale_diff/2.0))
            perturbed_contrast_state.edge_attr[:, 0] = \
                scale_factor*perturbed_contrast_state.edge_attr[:, 0]

            _, graph_embedding_1, proj_embedding_1 = model(
                state.x, state.edge_index, state.edge_attr,
                state.batch)

            with torch.no_grad():
                _, graph_embedding_2, proj_embedding_2 = ema_model(
                    perturbed_contrast_state.x,
                    perturbed_contrast_state.edge_index,
                    perturbed_contrast_state.edge_attr,
                    state.batch)

            loss = compute_loss(proj_embedding_1, proj_embedding_2)

            # accelerator.print(f"Loss: {loss.item()}")
            if accelerator:
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            epoch_loss += loss.item()

            if wandb_run:
                wandb_run.log({'training_step': num_steps, 'step_loss': loss.item()})
                wandb_run.log({'training_step': num_steps, 'lr': scheduler.get_lr()[0]})
            else:
                print(f"Step loss: {loss.item()}")

            if num_steps % config.update_freq == 0:
                ema_state_dict = ema_model.state_dict()
                for key, parameters in model.state_dict().items():
                    ema_state_dict[key] = config.ema_alpha*parameters \
                        + (1-config.ema_alpha)*parameters
                ema_model.load_state_dict(ema_state_dict)

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

        torch.save(model.state_dict(), config.save_model)
        if wandb_run:
            model_art = wandb.Artifact(config.model_name, type='model')
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
    parser.add_argument('--unit_actions_to_ignore', type=list, default=[],
                        help='Unit actions to ignore')
    parser.add_argument('--state_representation', type=str, default='graph',
                        help="State representation to use")
    parser.add_argument('--max_replays', default=-1, type=int,
                        help='Maximum number of replays to read')
    parser.add_argument('--max_replay_length', default=-1, type=int,
                        help='Maximum replay length')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')

    # Training Config
    parser.add_argument("--batch_size", default=8, type=int,
                        help='Batch size')
    parser.add_argument("--hidden_dims", default=128, type=int,
                        help='Hidden dim size')
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", default=4e-4, type=float,
                        help='Optimizer learning rate')
    parser.add_argument("--lr_exp_schedule_gamma", default=0.99, type=float,
                        help="Gamma value for exponential lr scheduler")
    parser.add_argument("--lr_warmup_steps", default=1000, type=int,
                        help="Number of warmup steps for cosine scheduler")
    parser.add_argument("--update_freq", default=100, type=int,
                        help="Frequency to update siamese network")
    parser.add_argument("--ema_alpha", default=0.99, type=float,
                        help="Alpha value for Exponential Moving Average (EMA)")
    parser.add_argument('--grad_accumulation_steps', default=4, type=int,
                        help="Number of steps to accumulate gradients")
    parser.add_argument('--mixed_precision', default=None, type=str,
                        help="Mixed-precision training")
    parser.add_argument('--device', default=None, type=str,
                        help="Device to run model and training")
    # Model Config
    parser.add_argument("--model_name", default="microrts-gnn-state-encoder", type=str,
                        help="Name of model")
    parser.add_argument("--save_model", default=None, type=str,
                        help="Filename for model")

    # Weights and Biases
    parser.add_argument('--project_name', default="microrts-graph-replay-encoder",
                        type=str, help="Name of project on W&Bs")
    parser.add_argument('--run_name', default="run-0",
                        type=str, help="Name of run on W&Bs")

    # Data Augmentation Config
    parser.add_argument("--min_scale_factor", default=0.8, type=float,
                        help="Minimum distance scaling factor")
    parser.add_argument("--max_scale_factor", default=1.2, type=float,
                        help="Maximum distance scaling factor")

    config = parser.parse_args()

    if config.device is None:
        config.device = torch.device(
            'cuda' if torch.cuda.is_available() \
                else 'mps' if torch.backends.mps.is_available() else 'cpu')


    config.alpha = [0.99, 0.99]
    config.theta = [0.99, 0.99]

    config.ignore_players = ['0']

    config.rng = torch.Generator(config.device).manual_seed(config.seed)

    return config

def main():
    """
    Main function
    """

    config = get_config()
    training_loop(config, debug_mode=False)

if __name__ == "__main__":
    main()
