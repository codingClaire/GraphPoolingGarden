import torch
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    GlobalAttention,
    Set2Set,
)
from pooling.sagpooling_layer import SAGPoolReadout
from pooling.diffpooling_layer import DiffPoolReadout


class PoolingLayer(torch.nn.Module):
    def __init__(self, params):
        ### Pooling function to generate whole-graph embeddings
        super(PoolingLayer, self).__init__()
        self.graph_pooling = params["pooling"]
        self.embed_dim = params["emb_dim"]

        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(
                    torch.nn.Linear(self.embed_dim, 2 * self.embed_dim),
                    torch.nn.BatchNorm1d(2 * self.embed_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * self.embed_dim, 1),
                )
            )
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(self.embed_dim, processing_steps=2)
        elif self.graph_pooling == "sagpool":
            self.pool = SAGPoolReadout(params["sagpool"], self.embed_dim)
        elif self.graph_pooling == "diffpool":
            self.pool = DiffPoolReadout(params["diffpool"],self.embed_dim)
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, x, batched_data):
        if self.graph_pooling == "sagpool":
            return self.pool(x, batched_data)
        return self.pool(x, batched_data.batch)