"""
Train action prediction model
"""

import argparse
from argparse import Namespace

import pandas as pd

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import random_split

from torch_geometric.loader import DataLoader

from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from accelerate.utils import set_seed

from diffusers import get_cosine_schedule_with_warmup

import wandb

from replay_parser.parser import parse_replay_dataset
from replay_parser.action import UNIT_ACTION_LIST

from nn.state_encoder import GNNStateEncoder
from nn.action_predictor import ActionPredictor

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
        _, players, _, map_id = filename.split('.')
        _, _, map_id = map_id.split('-')
        player_1, _, player_2, *_ = players.split('-')
        players = [player_1, player_2]

        if players[pid] in config.ignore_players:
            # Don't add player data to dataset
            continue

        for state, actions in trace:

            if node_dims == -1:
                node_dims = state.x.shape[-1]

            if edge_dims == -1:
                edge_dims = state.edge_attr.shape[-1]

            if actions is None:
                continue

            state.unit_actions = torch.tensor(state.unit_actions, dtype=torch.long)
            state.player_unit_mask = torch.tensor(state.player_unit_mask,
                                                  dtype=torch.bool)
            state.pid = pid
            state.map_id = map_id

            dataset.append(state)

    train_dataset, val_dataset = random_split(
        dataset, [config.train_val_split, 1.0-config.train_val_split],
        generator=config.rng)

    # assert len(train_dataset) == train_size
    # assert len(val_dataset) == val_size

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              generator=config.rng, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, generator=config.rng, shuffle=False)

    return train_loader, val_loader, node_dims, edge_dims

def create_model(node_dims: int, edge_dims: int,
                 hidden_dims: int, num_actions: int,
                 config: Namespace):
    """
    Create action prediction model

    :param node_dims: Number of nodes features
    :param edge_dims: Number of edge features
    :param hidden_dims: Hidden dimensions
    :param num_actions: Number of actions
    :param config: Script config
    :returns: Action prediction model
    """

    state_enc = GNNStateEncoder(node_dims, edge_dims, hidden_dims)
    if config.state_model is not None:
        model_dict = torch.load(config.state_model)
        state_enc.load_state_dict(model_dict)
        # Freeze model
        for param in state_enc.parameters():
            param.requires_grad = False

    action_predictor = ActionPredictor(state_enc, hidden_dims, num_actions)
    return action_predictor

def compute_loss(pred_logits: torch.Tensor, gt_unit_actions: torch.Tensor,
                 player_unit_mask: torch.Tensor):
    """
    Compute cross-entropy loss

    :param pred_logits: Predicted actions per unit
    :param gt_unit_actions: Ground truth actions per unit
    :param player_unit_mask: Mask for player units
    :returns: Cross-entropy loss 
    """

    pred_logits_ = pred_logits[player_unit_mask, :]
    gt_unit_actions_ = gt_unit_actions[player_unit_mask]

    # print(f"Pred actions: {pred_logits.shape} - {pred_logits_.shape}")
    # print(f"GT actions: {gt_unit_actions.shape} - {gt_unit_actions_.shape}")
    loss = F.cross_entropy(pred_logits_, gt_unit_actions_)

    return loss

@torch.no_grad()
def eval_loop(epoch: int, model, dataloader, wandb_run):
    """
    Evaluation loop
    """

    players = []
    maps = []
    accuracy = []
    # predictions = []
    # ground_truths = []

    avg_loss = 0
    avg_accuracy = 0
    for _, batch in enumerate(dataloader):

        pred_logits = model(batch)
        gt_unit_actions = batch.unit_actions
        player_unit_mask = batch.player_unit_mask

        pred_logits_ = pred_logits[player_unit_mask, :]
        gt_unit_actions_ = gt_unit_actions[player_unit_mask]
        preds = torch.argmax(pred_logits_, dim=-1)

        # print(f"Predictions: {preds} - Ground truth: {gt_unit_actions_}")

        loss = compute_loss(pred_logits, gt_unit_actions, player_unit_mask)
        avg_loss += loss.item()

        per_batch_accuracy = (preds == gt_unit_actions_).double().mean()
        # print(f"Per-batch accuracy: {per_batch_accuracy}")
        avg_accuracy += per_batch_accuracy

        players += batch.pid
        maps += batch.map_id
        accuracy += [per_batch_accuracy for _ in range(len(batch.pid))]
        # predictions += preds.tolist()
        # ground_truths += gt_unit_actions_.tolist()

    dataframe = {
        'player': players,
        'map': maps,
        'accuracy': accuracy
        # 'prediction': predictions,
        # 'ground_truth': ground_truths,
        }

    dataframe = pd.DataFrame(dataframe)
    dataframe['epoch'] = epoch

    avg_accuracy = avg_accuracy/len(dataloader)
    avg_loss = avg_loss/len(dataloader)
    # print(f"Average accuracy: {avg_accuracy}")

    if wandb_run:
        # table = wandb.Table(data=dataframe)
        wandb_run.log({'accuracy': avg_accuracy}, commit=False)
        wandb_run.log({'val-loss': avg_loss}, commit=False)
        # wandb_run.log({'val-table': table})
    else:
        print(f"Validation loss: {avg_loss}")
        print(f"Validation accuracy across {len(dataloader)} datapoints: {avg_accuracy}")

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

    train_dataloader, val_dataloader, node_dims, edge_dims = prepare_dataloader(config)
    action_pred_model = create_model(node_dims, edge_dims,
                                     config.hidden_dims, len(UNIT_ACTION_LIST),
                                     config)

    optimizer = optim.AdamW(action_pred_model.parameters(),
                            lr=config.learning_rate)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.lr_warmup_steps, len(train_dataloader)*config.num_train_epochs)

    if accelerator:
        action_pred_model, optimizer, train_dataloader, val_dataloader, scheduler \
            = accelerator.prepare(action_pred_model, optimizer,
                                  train_dataloader, val_dataloader, scheduler)

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
        action_pred_model.train()

        print(f"Epoch {epoch}")

        epoch_loss = 0
        num_iters = 0

        for _, batch in enumerate(train_dataloader):

            optimizer.zero_grad()

            pred_logits = action_pred_model(batch)
            gt_unit_actions = batch.unit_actions
            player_unit_mask = batch.player_unit_mask

            loss = compute_loss(pred_logits, gt_unit_actions, player_unit_mask)

            # accelerator.print(f"Loss: {loss.item()}")
            if accelerator:
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(action_pred_model.parameters(), 1.0)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(action_pred_model.parameters(), 1.0)

            epoch_loss += loss.item()

            if wandb_run:
                wandb_run.log({'training_step': num_steps, 'step_loss': loss.item()})
                wandb_run.log({'training_step': num_steps, 'lr': scheduler.get_lr()[0]})
            else:
                print(f"Step loss: {loss.item()}")

            num_steps += 1
            num_iters += 1

            # Update the model parameters with the optimizer
            optimizer.step()
            scheduler.step()

        # Validate model
        print("Evaluating model....")
        eval_loop(epoch, action_pred_model, val_dataloader, wandb_run)

        if wandb_run:
            wandb_run.log({'training_step': num_steps, 'epoch_loss': epoch_loss/num_iters})
        else:
            print(f"Epoch loss: {epoch_loss}")

    if config.save_model:
        # Save model to W&Bs

        torch.save(action_pred_model.state_dict(), config.save_model)
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
        description='Fine-Tune MicroRTS GNN State Encoder for Action Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_directory', help='Directory containing replays')

    # Replay Config
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

    # Training Config
    parser.add_argument("--batch_size", default=8, type=int,
                        help='Batch size')
    parser.add_argument("--hidden_dims", default=256, type=int,
                        help='Hidden dim size')
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", default=4e-3, type=float,
                        help='Optimizer learning rate')
    parser.add_argument("--lr_exp_schedule_gamma", default=0.99, type=float,
                        help="Gamma value for exponential lr scheduler")
    parser.add_argument("--lr_warmup_steps", default=500, type=int,
                        help="Number of warmup steps for cosine scheduler")
    parser.add_argument('--grad_accumulation_steps', default=4, type=int,
                        help="Number of steps to accumulate gradients")
    parser.add_argument('--mixed_precision', default=None, type=str,
                        help="Mixed-precision training")
    parser.add_argument('--device', default=None, type=str,
                        help="Device to run model and training")
    parser.add_argument('--debug_mode', action='store_true',
                        help="Flag to turn on debugging mode.")
    parser.add_argument('--train_val_split', default=0.9, type=float,
                        help="Percentage of dataset used for training.")

    # Model Config
    parser.add_argument("--model_name", default="action-predictor", type=str,
                        help="Name of model")
    parser.add_argument("--save_model", default=None, type=str,
                        help="Filename for model")
    parser.add_argument("--state_model", default=None,
                        help="Filename containing state model")

    # Weights and Biases
    parser.add_argument('--project_name', default="microrts-action-predictor",
                        type=str, help="Name of project on W&Bs")
    parser.add_argument('--run_name', default="run-0",
                        type=str, help="Name of run on W&Bs")

    config = parser.parse_args()

    if config.device is None:
        config.device = torch.device(
            'cuda' if torch.cuda.is_available() \
                else 'mps' if torch.backends.mps.is_available() else 'cpu')

    config.alpha = [0.99, 0.99]
    config.theta = [0.99, 0.99]

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
