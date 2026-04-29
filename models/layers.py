import torch
import torch.nn as nn
from torch_geometric.nn import (
    MessagePassing,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)


class NodeEncoder(nn.Module):
    def __init__(self, raw_node_feat_dim, embedding_dim, hidden_dim=None, num_layers=1):
        super().__init__()
        if num_layers == 0:
            self.encoder = nn.Identity()
        elif num_layers == 1:
            self.encoder = nn.Linear(raw_node_feat_dim, embedding_dim)
        else:
            if hidden_dim is None:
                hidden_dim = (raw_node_feat_dim + embedding_dim) // 2
            layers = [nn.Linear(raw_node_feat_dim, hidden_dim), nn.ReLU()]
            for _ in range(num_layers - 2):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            layers.append(nn.Linear(hidden_dim, embedding_dim))
            self.encoder = nn.Sequential(*layers)

    def forward(self, x_nodes):
        return self.encoder(x_nodes)


class GraphDecoder(nn.Module):
    def __init__(
        self,
        final_node_embedding_dim,
        mlp_hidden_dim,
        output_dim=2,
        pooling_type="mean",
        num_decoder_mlp_layers=3,
        dropout_rate=0.1,
    ):
        super().__init__()
        if pooling_type == "mean":
            self.pooling = global_mean_pool
        elif pooling_type == "add":
            self.pooling = global_add_pool
        elif pooling_type == "max":
            self.pooling = global_max_pool
        else:
            raise ValueError(f"Unsupported pooling type: {pooling_type}")

        layers = [
            nn.Linear(final_node_embedding_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        ]
        for _ in range(num_decoder_mlp_layers - 2):
            layers.extend(
                [
                    nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
        layers.append(nn.Linear(mlp_hidden_dim, output_dim))
        self.readout_mlp = nn.Sequential(*layers)

    def forward(self, final_node_embeddings, batch_vector):
        graph_embedding = self.pooling(final_node_embeddings, batch_vector)
        pred_output = self.readout_mlp(graph_embedding)
        return pred_output


class RichEdgeConv(MessagePassing):
    def __init__(
        self,
        node_feat_dim,
        edge_feat_dim,
        hidden_channels,
        out_channels,
        dropout_rate=0.2,
    ):
        super().__init__(aggr="mean")
        self.mlp_message = nn.Sequential(
            nn.Linear(node_feat_dim * 2 + edge_feat_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.mlp_update = nn.Sequential(
            nn.Linear(node_feat_dim + hidden_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_j, x_i, edge_attr], dim=1)
        return self.mlp_message(tmp)

    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=1)
        return self.mlp_update(tmp)
