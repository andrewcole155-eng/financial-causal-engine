import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATv2Conv
from torch_geometric.data import HeteroData
from typing import List, Union, Tuple


class MetaRelationTransformer(nn.Module):
    """Extends GAT to condition attention on specific KG meta-relations."""
    def __init__(self, metadata, hidden_dim):
        super().__init__()

        self.conv1 = HeteroConv(
            {
                edge_type: GATv2Conv(
                    (-1, -1),
                    hidden_dim,
                    heads=2,
                    concat=True,
                    add_self_loops=False
                )
                for edge_type in metadata[1]
            },
            aggr="sum"
        )

        self.conv2 = HeteroConv(
            {
                edge_type: GATv2Conv(
                    (hidden_dim * 2, hidden_dim * 2),
                    hidden_dim,
                    heads=1,
                    concat=False,
                    add_self_loops=False
                )
                for edge_type in metadata[1]
            },
            aggr="sum"
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: v.relu() for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: v.relu() for k, v in x_dict.items()}
        return x_dict


class MultiTaskTemporalGNN(nn.Module):
    def __init__(self, metadata, hidden_dim, num_risk_classes=3):
        super().__init__()

        self.spatial_encoder = MetaRelationTransformer(metadata, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)

        self.forecast_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, num_risk_classes)

        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        snapshots: Union[List[HeteroData], HeteroData]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if not isinstance(snapshots, list):
            snapshots = [snapshots]

        temporal_embeddings = []

        for day_data in snapshots:
            node_embs = self.spatial_encoder(
                day_data.x_dict,
                day_data.edge_index_dict
            )
            temporal_embeddings.append(node_embs["Company"])

        # (Nodes, Time, Features)
        seq_tensor = torch.stack(temporal_embeddings, dim=1)

        _, (hidden_state, _) = self.lstm(seq_tensor)
        final_embedding = hidden_state[-1]

        return (
            self.forecast_head(final_embedding),
            self.risk_head(final_embedding),
            torch.tensor([0.0], device=final_embedding.device),
        )


def create_hetero_model(data: HeteroData, hidden_dim=64) -> nn.Module:
    return MultiTaskTemporalGNN(data.metadata(), hidden_dim, num_risk_classes=3)
