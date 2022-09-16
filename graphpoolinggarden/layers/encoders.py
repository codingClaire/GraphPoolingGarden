import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class ASTFeatureEncoder(torch.nn.Module):
    def __init__(self, emb_dim, num_nodetypes, num_nodeattributes, max_depth):
        super(ASTFeatureEncoder, self).__init__()

        self.max_depth = max_depth
        self.type_encoder = torch.nn.Embedding(num_nodetypes, emb_dim)
        self.attribute_encoder = torch.nn.Embedding(num_nodeattributes, emb_dim)
        self.depth_encoder = torch.nn.Embedding(self.max_depth + 1, emb_dim)

    def forward(self, x, depth):
        depth[depth > self.max_depth] = self.max_depth
        return (
            self.type_encoder(x[:, 0])
            + self.attribute_encoder(x[:, 1])
            + self.depth_encoder(depth)
        )


class FeatureEncoder(torch.nn.Module):
    def __init__(
        self,
        dataset_name,
        in_dim,
        emb_dim
    ):
        super(FeatureEncoder, self).__init__()
        self.dataset = dataset_name
        if self.dataset == "ogbg-molhiv":
            self.feature_layer = AtomEncoder(emb_dim)
        else:
            self.feature_layer = nn.Linear(in_dim, emb_dim)

    def forward(self, x):
        return self.feature_layer(x)

class EdgeEncoder(torch.nn.Module):
    def __init__(self, dataset_name, in_dim, emb_dim):
        super(EdgeEncoder, self).__init__()
        if dataset_name == "ogbg-molhiv":
            self.feature_layer = BondEncoder(emb_dim)
        else:
            # TODO: adjust special features
            self.feature_layer = nn.Linear(in_dim, emb_dim)

    def forward(self, x):
        return self.feature_layer(x)
