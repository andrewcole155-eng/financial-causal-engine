# gnn_model.py

import torch
from torch_geometric.nn import GCNConv, SAGEConv, to_hetero
import torch.nn.functional as F

# 1. This is the simple, HOMOGENEOUS GNN block.
# It knows nothing about dictionaries or node types.
class GNNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = SAGEConv(in_channels, out_channels)

    # This is the "forward" function
    def forward(self, x, edge_index):
        # ⬇️ This line MUST be indented
        x = self.conv(x, edge_index).relu()
        # ⬇️ This line MUST be indented
        return x

# 2. This is the *main* model that `train.py` will use.
class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim):
        super().__init__()

        # --- Layer 1 ---
        self.gnn_layer1 = to_hetero(
            GNNBlock(-1, hidden_dim),  # -1 means auto-detect input features
            metadata,
            aggr='sum'
        )

        # --- Layer 2 ---
        self.gnn_layer2 = to_hetero(
            GNNBlock(hidden_dim, hidden_dim), # Input is now hidden_dim
            metadata,
            aggr='sum'
        )

        # --- Final Output Layer ---
        self.output_layer = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x_dict, edge_index_dict):

        # --- Run Layer 1 ---
        x_dict = self.gnn_layer1(x_dict, edge_index_dict)

        # --- Run Layer 2 ---
        x_dict = self.gnn_layer2(x_dict, edge_index_dict)

        # --- Get Company Output ---
        company_x = x_dict['Company']

        # --- Run Output Layer ---
        out = self.output_layer(company_x)

        return F.log_softmax(out, dim=1)


# 3. This is the "factory" function that `train.py` calls.
def create_hetero_model(data: 'HeteroData', hidden_dim=64, out_dim=3) -> torch.nn.Module:
    """
    A helper function to create our HeteroGNN model.
    """
    model = HeteroGNN(data.metadata(), hidden_dim, out_dim)
    return model