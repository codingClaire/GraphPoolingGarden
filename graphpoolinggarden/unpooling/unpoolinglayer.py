import torch
from unpooling.graphunet_unpooling_layer import GraphUnetUnpooling


class UnPoolingLayer(torch.nn.Module):
    def __init__(self, params):
        ### Pooling function to generate whole-graph embeddings
        super(UnPoolingLayer, self).__init__()
        self.unpool = params["unpooling"]
        self.embed_dim = params["emb_dim"]

        if self.unpool == "graphunetunpool":
            self.unpoolLayer = GraphUnetUnpooling()
        else:
            raise ValueError("Invalid unpooling type.")

    def forward(self, node_num, x, idxs):
        return self.unpoolLayer(node_num, x, idxs)
