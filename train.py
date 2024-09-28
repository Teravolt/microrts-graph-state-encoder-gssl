"""
Example usage of microrts replay parser
"""

import argparse
from argparse import Namespace

from copy import deepcopy

import numpy as np
import wandb

import torch
import torch.optim as optim

# from PIL import Image
# from PIL import ImageOps

from replay_parser.parser import parse_replay_dataset
# from replay_parser.state import MAX_HEIGHT, MAX_WIDTH

from nn.state_encoder import GNNStateEncoder

from torch_geometric.loader import DataLoader
# from torch_geometric.data import Dataset

def prepare_dataloader(config: Namespace):
    """
    Prepare dataloader
    """

    replay_data = parse_replay_dataset(config)

    input_dims = -1

    dataset = []
    for _, pid, trace in replay_data:
        for state, _ in trace:
            print(pid, state)
            contrast_state = deepcopy(state)
            if input_dims == -1:
                input_dims = state.x.shape[-1]
            contrast_state.x[:, -1] = (1 - contrast_state.x[:, -1])
            dataset.append((state, contrast_state))
    #         init_game_state = Image.fromarray(state[0].astype(np.uint8))
    #         init_game_state = ImageOps.fit(init_game_state, (64, 64))
    #         init_game_state.save(f'init-game-state-{filename}.png')

    #         raise
    return DataLoader(dataset, batch_size=config.batch_size), input_dims

def create_model(input_dims: int, hidden_dims: int, config: Namespace):
    """
    Create model
    """
    state_enc = GNNStateEncoder(input_dims, hidden_dims,
                                alpha=config.alpha, theta=config.theta)
    return state_enc

def compute_loss(output_1: torch.Tensor, output_2: torch.Tensor):
    """
    Compute contrastive loss
    """

    # Sum over each subset & average over each batch
    loss_fn = torch.nn.MSELoss(reduction='mean')
    # Cross entropy loss require (batch_size, x1 y1 x2 y2, ...)
    loss = loss_fn(output_1, output_2)
    return loss

# @torch.no_grad()
# def eval_loop(epoch: int, model: torch.nn.Module,
#               dataloader, wandb_run):
#     """
#     Evaluation loop
#     """

#     columns = ['pred']

#     dataframe = []
#     original_images = []
#     images = []
#     gt = []

#     avg_loss = 0
#     for i, batch in enumerate(dataloader):

#         logits = model(batch['image'])
#         labels = batch['label']

#         preds = torch.argmax(logits, dim=-1)
#         loss = compute_loss(logits, labels)
#         avg_loss += loss.item()

#         # acc = (preds == labels).double()
#         # print(f"Accuracy: {acc.mean().item()} - Val loss: {loss.item()}")
#         # wandb_run.log({'accuracy': acc.mean()}, commit=False)
#         # wandb_run.log({'val-loss': loss.item()}, commit=False)
        
#         for j in range(batch['image'].shape[0]):
#             images.append(batch['image'][j,:])
#             original_images.append(batch['original-image'][j,:])

#         dataframe += preds.tolist()
#         gt += batch['label'].tolist()

#         if i == 10:
#             break

#     dataframe = pd.DataFrame(dataframe,
#                              columns=columns)
#     dataframe['epoch'] = epoch
#     dataframe['image'] = images
#     dataframe['image'] = dataframe['image'].apply(tensor_to_pil)
#     dataframe['image'] = dataframe['image'].apply(wandb.Image)
#     dataframe['original_images'] = original_images
#     dataframe['original_images'] = dataframe['original_images'].apply(tensor_to_pil)
#     dataframe['original_images'] = dataframe['original_images'].apply(wandb.Image)

#     # dataframe['image'] = \
#     #     [wandb.Image(image) for image in images]
#     # dataframe['original_images'] = \
#     #     [wandb.Image(image) for image in original_images]
#     dataframe['gt'] = gt

#     # Get average accuracy and loss
#     acc = (dataframe['gt'] == dataframe['pred']).mean()
#     avg_loss = avg_loss/len(dataloader)

#     accelerator.print(
#         f"Val accuracy and loss: {acc} - {avg_loss}")

#     table = wandb.Table(data=dataframe)
#     wandb_run.log({'accuracy': acc}, commit=False)
#     wandb_run.log({'val-loss': loss}, commit=False)
#     wandb_run.log({'eval-table': table})

def training_loop(config: Namespace):
    """
    Training loop
    """

    # wandb_run = wandb.init(project='MicroRTS-Replay-Analyzer', entity=None,
    #                        job_type='training',
    #                        name=config.run_name,
    #                        config=config)

    # set_seed(config.seed)

    dataloader, input_dims = prepare_dataloader(config)
    model = create_model(input_dims, config.hidden_dims, config)
    ema_model = deepcopy(model)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

# #     scheduler = CosineAnnealingLR(
# #         optimizer,
# #         T_max=config.num_train_epochs)
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        config.lr_exp_schedule_gamma)

#     scheduler = CosineAnnealingWarmRestarts(
#         optimizer,
#         T_0=config.lr_warmup_steps)
        # last_epoch=config.num_train_epochs*len(train_dataloader))

    num_steps = 0
    for epoch in range(config.num_train_epochs):
        model.train()

        print(f"Epoch {epoch}")

        epoch_loss = 0
        num_iters = 0

        for _, batch in enumerate(dataloader):
    
            optimizer.zero_grad()
            input_1, input_2 = batch
            output_1 = model(input_1.x, input_1.edge_index, input_1.edge_attr)
            output_2 = ema_model(input_2.x, input_2.edge_index, input_2.edge_attr)

            loss = compute_loss(output_1, output_2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            epoch_loss += loss.item()

            # wandb_run.log({'loss': loss.item()}, commit=False, step=num_steps)
            # wandb_run.log({'lr': scheduler.get_lr()[0]}, commit=False, step=num_steps)

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

        # wandb_run.log({'epoch-loss': epoch_loss/num_iters})

    if config.save_model:
        # Save model to W&Bs
        # model_art = wandb.Artifact(config.model_name, type='model')
        torch.save(model.state_dict(), config.save_model)

        # model_art.add_file(config.save_model)
        # wandb_run.log_artifact(model_art)

    # wandb_run.finish()

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
    parser.add_argument("--learning_rate", default=128, type=float,
                        help='Optimizer learning rate')
    parser.add_argument("--lr_exp_schedule_gamma", default=0.99, type=float,
                        help="Gamma value for exponential lr scheduler")
    parser.add_argument("--update_freq", default=100, type=int,
                        help="Frequency to update siamese network")
    parser.add_argument("--ema_alpha", default=0.99, type=float,
                        help="Alpha value for Exponential Moving Average (EMA)")

    # Model Config
    parser.add_argument("--save_model", default=None, type=str,
                        help="Filename for model")

    config = parser.parse_args()

    config.model_name = "microrts-gnn-state-encoder"

    config.alpha = [0.99, 0.99]
    config.theta = [0.99, 0.99]

    return config

def main():
    """
    Main function
    """

    config = get_config()
    training_loop(config)

    # filenames = []
    # gnn_enc = GNNStateEncoder(12, 128, alpha=[0.99, 0.99], theta=[0.99, 0.99])
    # for filename, pid, trace in replay_data:
    #     gs = trace[0][0]
    #     out = gnn_enc(gs.x, gs.edge_index, gs.edge_attr)
    #     print(f"GCN output: {out.shape}")
    #     raise
    #     if filename not in filenames:
    #         filenames.append(filename)

            # init_game_state = Image.fromarray(trace[0][0][-1].astype(np.uint8))
    #         init_game_state = ImageOps.fit(init_game_state, (64, 64))
    #         init_game_state.save(f'init-game-state-{filename}.png')

if __name__ == "__main__":
    main()
