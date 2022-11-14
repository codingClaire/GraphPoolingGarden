import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from layers.encoders import EdgeEncoder


class GraphSAGEConv(MessagePassing):
    def __init__(self, dataset_name, edge_dim,emb_dim,
                aggr_name = "mean",normalize= False,root_weight= True,bias= True):
        super(GraphSAGEConv, self).__init__(aggr = aggr_name)
        self.edge_dim = edge_dim
        self.emb_dim = emb_dim 
        self.edge_encoder = EdgeEncoder(dataset_name, edge_dim,emb_dim)

        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(emb_dim , int):
            emb_dims  = (emb_dim , emb_dim)

        self.lin_l = nn.Linear(emb_dims[0], emb_dim, bias=bias)
        if self.root_weight:
            self.lin_r = nn.Linear(emb_dims[1], emb_dim, bias=False)
        
        self.reset_parameters()


    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()


    def forward(self, x, edge_index, edge_attr):
        if edge_attr != None and len(edge_attr.shape) != 1:
            edge_emb = self.edge_encoder(edge_attr)
        if isinstance(x, Tensor):
            x = (x, x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_emb)
        out = self.lin_l(out)

        if self.root_weight and x[1] is not None:
            out += self.lin_r(x[1])

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out
    
    def message(self, x_j, edge_attr):
        if edge_attr != None:
            return x_j + edge_attr
        else:
            return edge_attr

"""
class GraphSAGEConv(nn.Module):
    def __init__(self, dataset_name,edge_dim,emb_dim):
        super(GraphSAGEConv, self).__init__()
        self.emb_dim = emb_dim 
        self.edge_dim = edge_dim
        self.conv = SAGEConv(self.emb_dim,self.emb_dim)
    
    def forward(self, x,edge_index,edge_attr):
        return self.conv(x, edge_index)
"""
