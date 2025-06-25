"""
Evaluate action prediction model
"""

import argparse
from argparse import Namespace

import pandas as pd

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

from accelerate.utils import set_seed

import wandb

from replay_parser.parser import parse_replay_dataset
from replay_parser.action import UNIT_ACTION_LIST
from replay_parser.state import UNIT_TYPES

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

            # for i in range(len(UNIT_TYPES)):
            #     state.x[:, i] = 0
            # state.x[:, len(UNIT_TYPES)] = 0
            # state.x[:, len(UNIT_TYPES)+1] = 0

            dataset.append(state)

    dataloader = DataLoader(dataset, batch_size=1, generator=config.rng, shuffle=False)

    return dataloader, node_dims, edge_dims

def create_model(node_dims: int, _edge_dims: int,
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

    action_predictor = ActionPredictor(node_dims, hidden_dims, num_actions)

    model_dict = torch.load(config.action_model)
    action_predictor.load_state_dict(model_dict)

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
def eval_loop(config: Namespace, debug_mode=False):
    """
    Training loop

    :param config: Script config
    :param debug_mode: True if using debug mode
    """

    set_seed(config.seed)

    dataloader, node_dims, edge_dims = prepare_dataloader(config)
    action_pred_model = create_model(node_dims, edge_dims,
                                     config.hidden_dims, len(UNIT_ACTION_LIST),
                                     config)
    action_pred_model.eval()

    wandb_run = None
    if not debug_mode:
        wandb_run = wandb.init(project=config.project_name, entity=None,
                               job_type='evaluation',
                               name=config.run_name,
                               config=config)

        wandb_run.define_metric("eval_step")
        wandb_run.define_metric("eval_accuracy")
        wandb_run.define_metric("eval_loss")

        wandb_run.define_metric('eval_step_accuracy', step_metric='eval_step')
        wandb_run.define_metric('eval_step_loss', step_metric='eval_step')

    players = []
    maps = []
    accuracy = []

    avg_loss = 0
    avg_accuracy = 0
    for i, batch in enumerate(dataloader):

        pred_logits = action_pred_model(batch)
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

        if wandb_run:
            wandb_run.log({'eval_step': i, 'eval_step_loss': loss})
            wandb_run.log({'eval_step': i, 'eval_step_accuracy': per_batch_accuracy})

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

    avg_accuracy = avg_accuracy/len(dataloader)
    avg_loss = avg_loss/len(dataloader)
    # print(f"Average accuracy: {avg_accuracy}")

    if wandb_run:
        # table = wandb.Table(data=dataframe)
        wandb_run.log({'eval_accuracy': avg_accuracy}, commit=False)
        wandb_run.log({'eval_loss': avg_loss})
        # wandb_run.log({'val-table': table})
    else:
        print(f"Validation loss: {avg_loss}")
        print(f"Validation accuracy across {len(dataloader)} datapoints: {avg_accuracy}")

    if wandb_run:
        wandb_run.finish()

def get_config():
    """
    Build config

    :returns: Config from command line arguments
    """

    parser = argparse.ArgumentParser(
        description='Evaluate Action Prediction Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_directory', help='Directory containing replays')
    parser.add_argument("action_model", help="Filename containing state model")

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
    parser.add_argument('--frame_number_start', default=0, type=int,
                        help='Offset to start replay parsing')

    # Training Config
    parser.add_argument("--hidden_dims", default=256, type=int,
                        help='Hidden dim size')
    parser.add_argument('--device', default=None, type=str,
                        help="Device to run model and training")
    parser.add_argument('--debug_mode', action='store_true',
                        help="Flag to turn on debugging mode.")


    # Weights and Biases
    parser.add_argument('--project_name', default="microrts-action-predictor",
                        type=str, help="Name of project on W&Bs")
    parser.add_argument('--run_name', default="eval-0",
                        type=str, help="Name of run on W&Bs")

    config = parser.parse_args()

    if config.device is None:
        config.device = torch.device(
            'cuda' if torch.cuda.is_available() \
                else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Ignore RandomAI, POWorkerRush, POLightRush, and NaiveMCTS
    # These agents were used during training.
    config.ignore_players = ['0', '1', '2', '3']

    config.rng = torch.Generator(config.device).manual_seed(config.seed)

    return config

def main():
    """
    Main function
    """

    config = get_config()
    eval_loop(config, debug_mode=config.debug_mode)

if __name__ == "__main__":
    main()
