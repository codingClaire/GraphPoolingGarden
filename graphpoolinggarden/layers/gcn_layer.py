import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import degree

from layers.encoders import EdgeEncoder


class GCNConv(MessagePassing):
    def __init__(self, dataset_name, edge_dim, emb_dim):
        super(GCNConv, self).__init__(aggr="add")
        self.edge_dim = edge_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.edge_encoder = EdgeEncoder(dataset_name, edge_dim, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        row, col = edge_index

        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)  # D
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # norm for graph
        if edge_attr != None and len(edge_attr.shape) != 1:
            # for have edge_attr situation
            edge_embedding = self.edge_encoder(edge_attr)
            return self.propagate(
                edge_index, x=x, edge_attr=edge_embedding, norm=norm
            ) + F.relu(x + self.root_emb.weight) * 1.0 / deg.view(-1, 1)
        else:
            # for no edge_attr situation
            edge_embedding = 0
            return self.propagate(
                edge_index,
                x=x,
                norm=norm,
                edge_attr=edge_embedding,
                use_edge_attr=False,
            ) + F.relu(x + self.root_emb.weight) * 1.0 / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        if edge_attr != None:
            # for have edge_attr situation
            return norm.view(-1, 1) * F.relu(x_j + edge_attr)
        else:
            # for no edge_attr situation
            return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class GCNConvwithAdj(torch.nn.Module):
    # the edge_attr is not encoded in SingleGCNwithAdj
    def __init__(self, in_dim, emb_dim, drop_ratio, device, bias=True):
        super(GCNConvwithAdj, self).__init__()
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.in_dim = in_dim
        self.weight = torch.nn.Parameter(torch.FloatTensor(self.in_dim , self.emb_dim).to(device))
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(self.emb_dim).to(device))
        else:
            self.bias = None

    def forward(self, h, adj):
        h = F.dropout(h, self.drop_ratio, training = self.training)
        h = torch.matmul(adj, h)
        h = torch.matmul(h, self.weight)
        if self.bias is not None:
            h = h + self.bias
        return h