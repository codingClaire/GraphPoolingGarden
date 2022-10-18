import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool,global_add_pool

def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g

def top_k_graph(scores, edge_index, edge_attr, batch, h, k):
    """
    scores:y_tilde
    h: input_feature
    g: adj
    """
    num_nodes = batch.shape[0]
    # choose the topk nodes value and index
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    # update h
    new_h = torch.mul(new_h, values)
    # update adj (trick：user A^2 as the new adj)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    edge_index, edge_attr = transform_from_adj(un_g)
    batch = batch[idx]
    g = norm_g(un_g)
    return g, new_h, idx, edge_index, edge_attr, batch

def transform_from_adj(adj):
    # TODO 
    return adj,adj


class GraphUnetPool(nn.Module):

    def __init__(self, k,params, in_dim):
        super(GraphUnetPool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=params["drop_ratio"]) if params["drop_ratio"] > 0 else nn.Identity()

    def forward(self, h, edge_index, edge_attr, batch):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, edge_index, edge_attr, batch, h, self.k)


class GraphUnetReadout(nn.module):
    def __init__(self):
        super(GraphUnetReadout,self).__init__()
    
    def forward(self, graph_indicators, hs):
        for i in range(len(hs)):
            h_mean = global_mean_pool(hs[i], graph_indicators[i])
            h_sum = global_add_pool(hs[i], graph_indicators[i])
            h_max = global_max_pool(hs[i], graph_indicators[i])
        readout = torch.cat(h_max + h_sum + h_mean)
        return readout