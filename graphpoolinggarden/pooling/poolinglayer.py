import torch

from pooling.graphunet_pooling_layer import GraphUnetPool
from pooling.sagpooling_layer import SAGPoolReadout
from pooling.diffpooling_layer import DiffPoolReadout

class PoolingLayer(torch.nn.Module):
    def __init__(self, params,**kwargs):
        ### Pooling function to generate whole-graph embeddings
        super(PoolingLayer, self).__init__()
        self.graph_pooling = params["pooling"]
        self.embed_dim = params["emb_dim"]

        if self.graph_pooling == "graphunetpool":
            self.pool = GraphUnetPool(kwargs["k"],params["graphunetpool"],self.embed_dim)
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, x, batched_data):
        pool_list = ["sagpool","graphunetpool"]
        if self.graph_pooling in pool_list:                                                                                   
            return self.pool(x, batched_data)
        return self.pool(x, batched_data.batch)