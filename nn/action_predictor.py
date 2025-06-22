"""
MicroRTS action prediction models
"""

from torch import nn

class ActionPredictor(nn.Module):
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

        # print(f"Node embeddings: {node_embeddings.shape}")
        pred_logits = self.pred_layer(node_embeddings)

        return pred_logits
