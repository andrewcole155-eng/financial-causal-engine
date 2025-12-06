import torch
import torch.nn.functional as F
# UPDATE: We import GATv2Conv for the Attention mechanism
from torch_geometric.nn import SAGEConv, HeteroConv, GATv2Conv
from torch_geometric.data import HeteroData

# 1. The Main HeteroGNN Model
# We use GATv2 (Graph Attention) to allow the model to learn "sensitivity" (Edge Weights).
# This allows the model to decide that "Interest Rates" matter more for Tech than Energy.
class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim):
        super().__init__()
        
        # metadata[0] = node_types (e.g., ['Company', 'Event'])
        # metadata[1] = edge_types (e.g., [('Company', 'HAD_EVENT', 'Event'), ...])
        
        # --- Layer 1: Attention Layer ---
        # We replace SAGEConv with GATv2Conv.
        # - heads=1: Keeps output dimension equal to hidden_dim (simplifies stacking).
        # - add_self_loops=False: CRITICAL for Hetero graphs to prevent shape mismatch errors.
        self.conv1 = HeteroConv({
            edge_type: GATv2Conv((-1, -1), hidden_dim, heads=1, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='sum')

        # --- Layer 2: Refinement Layer ---
        self.conv2 = HeteroConv({
            edge_type: GATv2Conv((-1, -1), hidden_dim, heads=1, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='sum')

        # --- Final Output Layer ---
        # We only want to classify 'Company' nodes (Low/Med/High Risk)
        self.output_layer = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x_dict, edge_index_dict):
        # --- Run Layer 1 ---
        # The GAT layer automatically calculates attention weights (alphas) during this pass.
        x_dict = self.conv1(x_dict, edge_index_dict)
        
        # Apply ReLU to every node type in the dictionary
        x_dict = {key: x.relu() for key, x in x_dict.items()}

        # --- Run Layer 2 ---
        x_dict = self.conv2(x_dict, edge_index_dict)
        
        # Apply ReLU again
        x_dict = {key: x.relu() for key, x in x_dict.items()}

        # --- Get Company Output ---
        # We isolate the 'Company' nodes for the final prediction
        if 'Company' not in x_dict:
            # Safety check: if Company nodes got filtered out or lost
            raise ValueError("Company nodes missing from message passing output!")
            
        company_x = x_dict['Company']

        # --- Run Output Layer ---
        out = self.output_layer(company_x)

        # Return raw logits for CrossEntropyLoss
        return out

# 2. Factory function called by train.py
def create_hetero_model(data: HeteroData, hidden_dim=64, out_dim=3) -> torch.nn.Module:
    """
    Helper function to initialize the HeteroGNN.
    """
    # Pass metadata (node types and edge types) so the model knows what layers to build
    model = HeteroGNN(data.metadata(), hidden_dim, out_dim)
    return model