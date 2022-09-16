import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F

from layers.encoders import EdgeEncoder


class GINConv(MessagePassing):
    def __init__(self, dataset_name, edge_dim,emb_dim):
        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential( torch.nn.Linear(emb_dim, 2*emb_dim), 
                                        torch.nn.BatchNorm1d(2*emb_dim), 
                                        torch.nn.ReLU(), 
                                        torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.edge_encoder = EdgeEncoder(dataset_name, edge_dim,emb_dim)
        # self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr != None:
            # for have edge_attr situation  
            edge_embedding = self.edge_encoder(edge_attr)
            out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding, use_edge_attr = True))
        else:
            # for no edge_attr situation 
            edge_embedding = 0 # not really use
            out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding, use_edge_attr = False))
        return out

    def message(self, x_j, edge_attr,use_edge_attr):
        if use_edge_attr != False:
            # for have edge_attr situation  
            return F.relu(x_j + edge_attr)
        else:
            # for no edge_attr situation  
            return F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out