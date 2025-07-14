"""
MicroRTS State Encoders
"""

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv

class GNNStateEncoder(nn.Module):
    """
    Encode state using GNN
    """

    def __init__(self, node_in_dims: int, edge_in_dims: int, hidden_dims: int) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(node_in_dims)
        self.node_linear_1 = nn.Linear(node_in_dims, hidden_dims)

        self.gnn_layer_1 = GATv2Conv(
            hidden_dims,
            hidden_dims,
            edge_dim=edge_in_dims)
        self.gnn_layer_2 = GATv2Conv(
            hidden_dims,
            hidden_dims,
            edge_dim=edge_in_dims)
        # self.gnn_layer_3 = GATv2Conv(
        #     hidden_dims,
        #     hidden_dims,
        #     edge_dim=edge_in_dims)
        # self.gnn_layer_4 = GATv2Conv(
        #     hidden_dims,
        #     hidden_dims,
        #     edge_dim=edge_in_dims)

        self.graph_projection_1 = nn.Linear(hidden_dims, hidden_dims)
        self.graph_projection_2 = nn.Linear(hidden_dims, hidden_dims)

        # self.node_projection_1 = nn.Linear(hidden_dims, hidden_dims)
        # self.node_projection_2 = nn.Linear(hidden_dims, hidden_dims)

    def forward(self, x: torch.Tensor,
                edge_indices: torch.Tensor,
                edge_weights: torch.Tensor,
                batch: torch.Tensor=None):
        """
        Forward pass
        """

        node_out = self.layer_norm(x)
        # print(f"Input dimensions: {x.shape}")
        # node_out = self.layer_norm(x)
        node_out = self.node_linear_1(node_out)
        node_out = F.relu(node_out)
        # edge_out = self.edge_linear_1(edge_weights)

        # print(f"Node output: {node_out.shape}")

        gnn_out_1 = self.gnn_layer_1(x=node_out,
                                     edge_index=edge_indices,
                                     edge_attr=edge_weights)
        # print(f"GNN layer output 1: {gnn_out_1.shape}")

        gnn_out_2 = self.gnn_layer_2(x=gnn_out_1,
                                     edge_index=edge_indices,
                                     edge_attr=edge_weights)

        # gnn_out_3 = self.gnn_layer_3(x=gnn_out_2,
        #                              edge_index=edge_indices,
        #                              edge_attr=edge_weights)

        # gnn_out_4 = self.gnn_layer_4(x=gnn_out_3,
        #                              edge_index=edge_indices,
        #                              edge_attr=edge_weights)

        # print(f"GNN layer output 2: {gnn_out_2.shape}")

        graph_embedding = None
        for i in range(batch.max()+1):
            indices = torch.where(batch == i)[0]
            # print(indices, gnn_out_2[indices, :].shape)
            mean = torch.mean(gnn_out_2[indices, :], dim=0).unsqueeze(0)
            graph_embedding = mean if graph_embedding is None \
                else torch.concat([graph_embedding, mean], dim=0)

        graph_proj_embedding = F.relu(self.graph_projection_1(graph_embedding))
        graph_proj_embedding = self.graph_projection_2(graph_proj_embedding)

        # node_proj_embedding = F.relu(self.node_projection_1(gnn_out_2))
        # node_proj_embedding = self.node_projection_2(node_proj_embedding)

        return gnn_out_2, graph_embedding, graph_proj_embedding


class GNNStateEncoderV2(nn.Module):
    """
    Encode state using GNN
    """

    def __init__(self, node_in_dims: int, edge_in_dims: int, hidden_dims: int) -> None:
        super().__init__()

        self.gnn_layer_1 = GATv2Conv(
            node_in_dims,
            hidden_dims,
            edge_dim=edge_in_dims)
        self.linear_1 = nn.Linear(node_in_dims, hidden_dims)

        self.gnn_layer_2 = GATv2Conv(
            hidden_dims,
            hidden_dims,
            edge_dim=edge_in_dims)
        self.linear_2 = nn.Linear(hidden_dims, hidden_dims)

        # self.gnn_layer_3 = GATv2Conv(
        #     2*hidden_dims,
        #     hidden_dims,
        #     edge_dim=edge_in_dims)
        # self.linear_3 = nn.Linear(2*hidden_dims, hidden_dims)

        self.graph_projection_1 = nn.Linear(hidden_dims, hidden_dims)
        self.graph_projection_2 = nn.Linear(hidden_dims, hidden_dims)

        # self.node_projection_1 = nn.Linear(hidden_dims, hidden_dims)
        # self.node_projection_2 = nn.Linear(hidden_dims, hidden_dims)

    def forward(self, x: torch.Tensor,
                edge_indices: torch.Tensor,
                edge_weights: torch.Tensor,
                batch: torch.Tensor=None):
        """
        Forward pass
        """

        gnn_1_out = self.gnn_layer_1(
            x=x, edge_index=edge_indices,
            edge_attr=edge_weights)
        linear_1_out = self.linear_1(x)

        output_1 = F.elu(gnn_1_out + linear_1_out)
        # print(f"GNN layer output 1: {output_1.shape}")

        gnn_2_out = self.gnn_layer_2(
            x=output_1, edge_index=edge_indices,
            edge_attr=edge_weights)
        linear_2_out = self.linear_2(output_1)

        output_2 = gnn_2_out + linear_2_out
        # print(f"GNN layer output 2: {output_2.shape}")

        # gnn_3_out = self.gnn_layer_3(
        #     x=output_2, edge_index=edge_indices,
        #     edge_attr=edge_weights)
        # linear_3_out = self.linear_3(output_2)

        # output_3 = gnn_3_out + linear_3_out
        # print(f"GNN layer output 3: {output_3.shape}")

        graph_embedding = None
        for i in range(batch.max()+1):
            indices = torch.where(batch == i)[0]
            # print(indices, gnn_out_2[indices, :].shape)
            mean = torch.mean(output_2[indices, :], dim=0).unsqueeze(0)
            graph_embedding = mean if graph_embedding is None \
                else torch.concat([graph_embedding, mean], dim=0)

        graph_proj_embedding = F.relu(self.graph_projection_1(graph_embedding))
        graph_proj_embedding = self.graph_projection_2(graph_proj_embedding)

        # node_proj_embedding = F.relu(self.node_projection_1(gnn_out_2))
        # node_proj_embedding = self.node_projection_2(node_proj_embedding)

        return output_2, graph_embedding, graph_proj_embedding

# class TokenGridStateEncoder(nn.Module):
#     """
#     Grid-based state encoder
#     """

#     def __init__(self):
#         super().__init__()

#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
#         self.text_embedding_model = AutoModelForMaskedLM.from_pretrained(model_id)

#     def forward(self, x: torch.Tensor):
#         """
#         Forward pass
#         """

#         output = self.text_embedding_model(x)
#         print(f"Output: {output.shape}")
#         raise

class MLPStateEncoder(nn.Module):
    """
    Encode state using MLP
    """

    def __init__(self, in_dims: int, hidden_dims: int) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(in_dims, hidden_dims)

    def forward(self, x: torch.Tensor):
        """
        Forward pass
        """

        return self.linear_1(x)
