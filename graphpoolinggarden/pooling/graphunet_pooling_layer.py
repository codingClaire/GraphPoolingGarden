import torch
import torch.nn as nn
from torch_scatter import scatter

def top_k_graph(scores, edge_index, edge_attr, batch, h, k):
    """
    scores:y_tilde
    h: input_feature
    g: adj
    """
    num_nodes = batch.shape[0]
    # choose the topk nodes value and index
    values, idx = torch.topk(scores, max(2, int(k * num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    # update h
    new_h = torch.mul(new_h, values)
    # update adj (trick：user A^2 as the new adj)

    edge_attr = torch.ones([edge_index.shape[1]]).to(batch.device)
    g = torch.sparse.FloatTensor(
        edge_index, edge_attr, torch.Size([batch.shape[0], batch.shape[0]])
    )
    g = g.to_dense()
    # the source code use bool value, while in order to preserve edge_attr
    # the bool operation is removed in here
    un_g = g.float()
    un_g = torch.matmul(un_g, un_g).float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]

    edge_index = torch.nonzero(un_g).T
    # edge_attr = un_g[edge_index[0],edge_index[1]]

    batch = batch[idx]
    return new_h, edge_index, edge_attr, batch, idx


class GraphUnetPool(nn.Module):
    def __init__(self, k, params, in_dim):
        super(GraphUnetPool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = (
            nn.Dropout(p=params["drop_ratio"])
            if params["drop_ratio"] > 0
            else nn.Identity()
        )

    def forward(self, h, edge_index, edge_attr, batch):
        # calculate the score
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        # find the top_k nodes for next operation
        # for Graph U-net, the original model didn't use edge_attr
        return top_k_graph(scores, edge_index, edge_attr, batch, h, self.k)


class GraphUnetReadout(nn.Module):
    def __init__(self):
        super(GraphUnetReadout, self).__init__()

    def forward(self, hs, graph_indicators):
        layer = len(graph_indicators)
        batch, dim = int(max(graph_indicators[0]) + 1), hs[0].shape[1]
        #preserve_graph_indicators = [
        #    torch.unique(graph_indicator).to(hs[0].device) for graph_indicator in graph_indicators
        #]
        #shapes = [
        #    preserve_graph_indicator.shape[0]
        #    for preserve_graph_indicator in preserve_graph_indicators
        #]
        h_mean, h_sum, h_max = [], [], []  # torch.zeros_like(batch,layer*dim)
        for i in range(layer):
            #if shapes[layer - i - 1] != batch:
            h_mean.append(scatter(hs[i], graph_indicators[layer - i - 1], dim=0, dim_size=batch, reduce='mean'))
            h_sum.append(scatter(hs[i], graph_indicators[layer - i - 1], dim=0, dim_size=batch, reduce='add'))
            h_max.append(scatter(hs[i], graph_indicators[layer - i - 1], dim=0, dim_size=batch, reduce='max'))
            #else:
            #    h_mean.append(global_mean_pool(hs[i], graph_indicators[layer - i - 1]))
            #    h_sum.append(global_add_pool(hs[i], graph_indicators[layer - i - 1]))
            #    h_max.append(global_max_pool(hs[i], graph_indicators[layer - i - 1]))
        readout = torch.cat(
            (
                torch.cat(h_max, dim=1),
                torch.cat(h_sum, dim=1),
                torch.cat(h_mean, dim=1),
            ),
            dim=1,
        )
        return readout
