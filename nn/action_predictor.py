"""
MicroRTS action prediction models
"""

from torch import nn
import torch.nn.functional as F

class ActionPredictorGNN(nn.Module):
    """
    Linear action prediction model
    """

    def __init__(self, state_encoder: nn.Module,
                 hidden_dims: int, out_dims: int) -> None:
        super().__init__()

        self.state_encoder = state_encoder
        self.pred_layer = nn.Linear(hidden_dims, out_dims)

    def forward(self, state):
        """
        Forward pass
        """

        node_embeddings, _, _ = self.state_encoder(
            state.x,
            state.edge_index,
            state.edge_attr,
            state.batch)

        pred_logits = self.pred_layer(node_embeddings)

        return pred_logits

class ActionPredictor(nn.Module):
    """
    Linear action prediction model
    """

    def __init__(self, node_in_dims: int,
                 hidden_dims: int, out_dims: int) -> None:
        super().__init__()

        self.layer_1 = nn.Linear(node_in_dims, hidden_dims)
        self.layer_2 = nn.Linear(hidden_dims, hidden_dims)
        self.layer_3 = nn.Linear(hidden_dims, hidden_dims)
        self.pred_layer = nn.Linear(hidden_dims, out_dims)

    def forward(self, state):
        """
        Forward pass
        """

        layer_1_out = F.relu(self.layer_1(state.x))
        layer_2_out = F.relu(self.layer_2(layer_1_out))
        layer_3_out = F.relu(self.layer_3(layer_2_out))
        pred_logits = self.pred_layer(layer_3_out)

        return pred_logits
