import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder

class FeatureEncoder(torch.nn.Module):
    def __init__(self,dataset_name,in_dim,emb_dim):
        super(FeatureEncoder, self).__init__()
        if dataset_name == "ogbg-molhiv":
            self.feature_layer = AtomEncoder(emb_dim)
        else:
            # TODO: adjust special features
            self.feature_layer = nn.Linear(in_dim,emb_dim)

    def forward(self, x):
        return self.feature_layer(x)


class EdgeEncoder(torch.nn.Module):
    def __init__(self,dataset_name,in_dim, emb_dim):
        super(EdgeEncoder, self).__init__()
        if dataset_name == "ogbg-molhiv":
            self.feature_layer = BondEncoder(emb_dim)
        else:
            # TODO: adjust special features
            self.feature_layer = nn.Linear(in_dim,emb_dim)

    def forward(self, x):
        return self.feature_layer(x)