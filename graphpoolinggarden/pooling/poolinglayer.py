import torch

from pooling.graphunet_pooling_layer import GraphUnetPool

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

    def forward(self, h, edge_index, edge_attr, batch):
        if self.graph_pooling == "graphunetpool":                                                                                   
            return self.pool(h, edge_index,edge_attr, batch)  
        return self.pool(h,batch)