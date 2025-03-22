"""
Replay Encoder
"""

import torch
from torch import nn

from nn.state_encoder import GNNStateEncoder

class GNNReplayEncoder(nn.Module):
    """
    Encode replay using GNN state encoder
    """

    def __init__(self, node_in_dims: int, edge_in_dims: int, hidden_dims: int) -> None:
        super().__init__()

        self.state_encoder = GNNStateEncoder(
            node_in_dims, edge_in_dims, hidden_dims)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dims, nhead=1)
        self.replay_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, node_features: torch.Tensor,
                edge_indices: torch.Tensor,
                edge_weights: torch.Tensor,
                batch: torch.Tensor=None):
        """
        Forward pass
        """

        _, latent_states, _ = self.state_encoder(
            node_features,
            edge_indices,
            edge_weights,
            batch)

        latent_states = latent_states.unsqueeze(0)

        print(f"Latent states of nodes: {latent_states.shape}")

        output = self.replay_encoder(latent_states)

        print(f"Output shape: {output.shape}")

        return output
