import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATv2Conv
from torch_geometric.data import HeteroData
from typing import List, Union, Tuple

# ---------------------------------------------------------
# COMPONENT 1: SPATIAL ENCODER (Unchanged)
# ---------------------------------------------------------
class SpatialGAT(nn.Module):
    def __init__(self, metadata, hidden_dim):
        super().__init__()
        self.conv1 = HeteroConv({
            edge_type: GATv2Conv((-1, -1), hidden_dim, heads=1, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='sum')

        self.conv2 = HeteroConv({
            edge_type: GATv2Conv((-1, -1), hidden_dim, heads=1, add_self_loops=False)
            for edge_type in metadata[1]
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        return x_dict

# ---------------------------------------------------------
# COMPONENT 2: MULTI-TASK TEMPORAL PREDICTOR
# ---------------------------------------------------------
class MultiTaskTemporalGNN(nn.Module):
    def __init__(self, metadata, hidden_dim, num_risk_classes=3):
        super().__init__()
        
        # Shared Layers (The Brain)
        self.spatial_encoder = SpatialGAT(metadata, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=1, 
                            batch_first=True)
        
        # HEAD 1: Forecasting (Regression) -> Output: 1 value (Price Return)
        self.forecast_head = nn.Linear(hidden_dim, 1)
        
        # HEAD 2: Risk (Classification) -> Output: 3 values (Logits for Low/Med/High)
        self.risk_head = nn.Linear(hidden_dim, num_risk_classes)

    def forward(self, snapshots: Union[List[HeteroData], HeteroData]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(snapshots, list):
            snapshots = [snapshots]

        temporal_embeddings = []

        # 1. Spatial Pass
        for day_data in snapshots:
            all_node_embs = self.spatial_encoder(day_data.x_dict, day_data.edge_index_dict)
            if 'Company' not in all_node_embs:
                raise ValueError("Company nodes missing!")
            temporal_embeddings.append(all_node_embs['Company'])

        # 2. Temporal Pass
        seq_tensor = torch.stack(temporal_embeddings, dim=0).permute(1, 0, 2)
        _, (hidden_state, _) = self.lstm(seq_tensor)
        final_embedding = hidden_state[-1] 

        # 3. Multi-Head Output
        # Forecast: Continuous value
        forecast_out = self.forecast_head(final_embedding) 
        
        # Risk: Class logits
        risk_out = self.risk_head(final_embedding) 
        
        return forecast_out, risk_out

def create_hetero_model(data: HeteroData, hidden_dim=64, out_dim=None) -> nn.Module:
    # out_dim is ignored here as we hardcode the 2 heads
    model = MultiTaskTemporalGNN(data.metadata(), hidden_dim, num_risk_classes=3)
    return model