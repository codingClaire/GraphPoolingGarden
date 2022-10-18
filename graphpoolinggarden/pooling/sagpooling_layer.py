import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.nn import global_mean_pool, global_max_pool


class SAGPool(torch.nn.Module):
    """SelfAttentionGraphPooling"""

    def __init__(self, params, embed_dim):
        super(SAGPool, self).__init__()
        self.input_dim = embed_dim
        self.keep_ratio = params["keep_ratio"]
        if params["activation"] == "tanh":
            self.activation = torch.tanh
        # for calculate the attention score
        self.attention_layer = GCNConv(self.input_dim, 1)

    def forward(self, input_feature, graph_indicator, edge_index, edge_attr=None):
        # use gcn to learn attention score
        # attn_score = self.attention_layer(adjacency, input_feature).squeeze()
        attn_score = self.attention_layer(input_feature, edge_index).squeeze()
        # find the topk as nextk
        mask = topk(attn_score, self.keep_ratio, graph_indicator)
        new_input_feature = input_feature[mask] * self.activation(
            attn_score[mask]
        ).view(-1, 1)
        new_graph_indicator = graph_indicator[mask]
        # get new graph
        # new_adjacency = filter_adjacency(adjacency, mask)
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, mask, num_nodes=attn_score.size(0)
        )
        return new_input_feature, edge_index, edge_attr, new_graph_indicator  # , mask

# TODO: change the SAGPoolReadout
class SAGPoolReadout(torch.nn.Module):
    def __init__(self, sagpool_params, embed_dim):
        super(SAGPoolReadout, self).__init__()
        self.sagpool_params = sagpool_params
        self.embed_dim = embed_dim
        self.layer_num = sagpool_params["layer_num"]
        self.pools = torch.nn.ModuleList()
        for _ in range(self.layer_num):
            self.pools.append(SAGPool(sagpool_params, embed_dim))
        self.linear = torch.nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, input_feature, batched_data):
        """input the graph information and output the readout result"""
        edge_index = batched_data.edge_index
        edge_attr = batched_data.edge_attr
        graph_indicator = batched_data.batch
        for l in range(self.layer_num):
            input_feature, edge_index, edge_attr, graph_indicator = self.pools[l](
                input_feature, graph_indicator, edge_index, edge_attr
            )
        readout = torch.cat(
            [
                global_mean_pool(input_feature, graph_indicator),
                global_max_pool(input_feature, graph_indicator),
            ],
            dim=1,
        )
        return self.linear(readout)
