"""
MicroRTS State Encoders
"""

import torch
from torch import nn

from torch_geometric.nn import GCN2Conv

class GNNStateEncoder(nn.Module):
    
    def __init__(self, in_dims: int, hidden_dims: int,
                 alpha: list[float], theta: list[float]) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(in_dims, hidden_dims)

        self.gcn_1 = GCN2Conv(hidden_dims, alpha=alpha[0],
                              theta=theta[0], layer=1)
        self.gcn_2 = GCN2Conv(hidden_dims, alpha=alpha[1],
                              theta=theta[1], layer=2)

    def forward(self, x: torch.Tensor,
                edge_indices: torch.Tensor,
                edge_weights: torch.Tensor):
        """
        Forward pass
        """

        linear_out = self.linear_1(x)
        print(f"Linear output: {linear_out.shape}")

        gcn1_out = self.gcn_1(x=linear_out,
                              x_0=linear_out,
                              edge_index=edge_indices,
                              edge_weight=edge_weights)
        print(f"GCN1 output: {gcn1_out.shape}")

        gcn2_out = self.gcn_2(x=gcn1_out,
                              x_0=linear_out,
                              edge_index=edge_indices,
                              edge_weight=edge_weights)

        print(f"GCN2 output: {gcn2_out.shape}")
        return gcn2_out

class MLPStateEncoder():
    
    def __init__(self, in_dims: int, hidden_dims: int) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(in_dims, hidden_dims)

    def forward(self, x: torch.Tensor):
        """
        Forward pass
        """
