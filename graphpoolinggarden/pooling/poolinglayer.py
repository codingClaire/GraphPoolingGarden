import torch
from pooling.diffpooling_layer import DiffPool
from pooling.graphunet_pooling_layer import GraphUnetPool


class PoolingLayer(torch.nn.Module):
    def __init__(self, params, **kwargs):
        ### Pooling function to generate whole-graph embeddings
        super(PoolingLayer, self).__init__()
        self.graph_pooling = params["pooling"]
        self.embed_dim = params["emb_dim"]

        #if self.graph_pooling == "diffpool":
        #    self.hidden_dim = self.embed_dim if "hidden_dim" not in kwargs.keys() else kwargs["hidden_dim"]
        #    self.pool = DiffPool(params["diffpool"], self.hidden_dim, self.embed_dim)
        if self.graph_pooling == "graphunetpool":
            self.pool = GraphUnetPool(
                kwargs["k"], params["graphunetpool"], self.embed_dim
            )
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, h, edge_index, edge_attr, batch):
        if self.graph_pooling == "graphunetpool":
            # notice that in graph u-net pooling, the edge_attr is not used
            return self.pool(h, edge_index, edge_attr, batch)
        #elif self.graph_pooling == "diffpool":
        #    return self.pool(h, adj)
        else:
            return self.pool(h, batch)
