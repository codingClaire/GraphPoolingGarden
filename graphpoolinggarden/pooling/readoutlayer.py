import torch
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    GlobalAttention,
    Set2Set,
)
from pooling.sagpooling_layer import SAGPoolReadout
from pooling.graphunet_pooling_layer import GraphUnetReadout

class ReadoutLayer(torch.nn.Module):
    def __init__(self, params,**kwargs):
        ### Pooling function to generate whole-graph embeddings
        super(ReadoutLayer, self).__init__()
        self.graph_pooling = params["pooling"]
        self.embed_dim = params["emb_dim"]

        if self.graph_pooling == "sum":
            self.readout = global_add_pool
        elif self.graph_pooling == "mean":
            self.readout = global_mean_pool
        elif self.graph_pooling == "max":
            self.readout = global_max_pool
        elif self.graph_pooling == "attention":
            self.readout = GlobalAttention(
                gate_nn=torch.nn.Sequential(
                    torch.nn.Linear(self.embed_dim, 2 * self.embed_dim),
                    torch.nn.BatchNorm1d(2 * self.embed_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * self.embed_dim, 1),
                )
            )
        elif self.graph_pooling == "set2set":
            self.readout = Set2Set(self.embed_dim, processing_steps=2)
        elif self.graph_pooling == "sagpool":
            self.readout = SAGPoolReadout(params["sagpool"], self.embed_dim)
        elif self.graph_pooling == "graphunetpool":
            self.readout = GraphUnetReadout()
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, hs, gs):
        if self.graph_pooling == "graphunetpool":
            return self.readout(hs,gs)
        else:
            return self.readout(hs,gs.batch)