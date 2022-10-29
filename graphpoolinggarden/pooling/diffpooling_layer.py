import torch
import torch.nn as nn
from torch.nn import init
from torch_scatter import scatter

from layers.gcn_layer import GCNConv
from layers.GNN_node import GnnLayerwithAdj


class EntropyLoss(nn.Module):
    # Return Scalar
    def forward(self, adj, anext, s_l):
        entropy = (
            (torch.distributions.Categorical(probs=s_l).entropy()).sum(-1).mean(-1)
        )
        assert not torch.isnan(entropy)
        return entropy


class LinkPredLoss(nn.Module):
    def forward(self, adj, anext, s_l):
        link_pred_loss = (adj - s_l.matmul(s_l.transpose(-1, -2))).norm(dim=(1, 2))
        link_pred_loss = link_pred_loss / (adj.size(1) * adj.size(2))
        return link_pred_loss.mean()


class DiffPool(nn.Module):
    def __init__(self, params, in_dim, assign_dim,**kwargs):
        super(DiffPool, self).__init__()
        self.max_num_nodes = params["max_num_nodes"]
        # self.assign_hidden_dim = params["assign_hidden_dim"]
        self.assign_hidden_dim = assign_dim
        self.assign_ratio = params["assign_ratio"]
        self.assign_num_layers = params["assign_num_layers"]
        self.two_dim = kwargs["two_dim"] if "two_dim" in kwargs.keys() else False
        self.concat = params["concat"]
        self.bias = params["bias"]
        self.add_self = not self.concat
        if params["activation"] == "ReLU":
            self.activation = nn.ReLU()

        #### assignLayers ####
        assign_params = params
        assign_params["emb_dim"] = assign_dim  # self.assign_hidden_dim
        assign_params["num_layer"] = self.assign_num_layers

        if self.concat:
            assign_params["JK"] = "concat"
            assign_pred_input_dim = (
                self.assign_hidden_dim * (self.assign_num_layers - 1) + assign_dim
            )
        else:
            assign_params["JK"] = "last"
            assign_pred_input_dim = assign_dim

        # assignment(GNN_l,pool)
        self.assignLayers = GnnLayerwithAdj(assign_params, two_dim= self.two_dim, in_dim=in_dim)
        self.assignPredLayer = nn.Linear(assign_pred_input_dim, assign_dim)

        self.init_weights()

    def forward(self, h, adj):
        # equation 5 (GNN_l,pool)
        self.assign_tensor = self.assignLayers(h, adj)
        # equation 6
        self.assign_tensor = self.assignPredLayer(self.assign_tensor)
        self.assign_tensor = nn.Softmax(dim=-1)(self.assign_tensor)
        # update pooled feature and adj max
        h = torch.matmul(torch.transpose(self.assign_tensor, -1, -2), h)
        adj = (
            torch.transpose(self.assign_tensor, -1, -2)
            @ adj
            @ self.assign_tensor
        )
        # TODO: calculate the loss
        return h, adj

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, GCNConv):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)


"""
class BatchedDiffPool(nn.Module):
    def __init__(self, params, in_dim, assign_dim):
        super(BatchedDiffPool, self).__init__()
        self.max_num_nodes = params["max_num_nodes"]
        self.assign_hidden_dim = assign_dim  #  params["assign_hidden_dim"]
        self.assign_ratio = params["assign_ratio"]
        self.assign_num_layers = params["assign_num_layers"]
        self.concat = params["concat"]
        self.bias = params["bias"]
        self.add_self = not self.concat
        if params["activation"] == "ReLU":
            self.activation = nn.ReLU()

        # assignment(GNN_l,pool)
        self.assignLayers = torch.nn.ModuleList()
        self.assignPredLayers = torch.nn.ModuleList()

        #### assignLayers ####
        assign_params = params
        assign_params["in_dim"] = in_dim
        assign_params["emb_dim"] = assign_dim  # self.assign_hidden_dim
        assign_params["num_layer"] = self.assign_num_layers

        self.assignLayers.append(GnnLayerwithAdj(assign_params))
        #### assignPredLayers ####
        if self.concat:
            assign_pred_input_dim = (
                self.assign_hidden_dim * (self.assign_num_layers - 1) + assign_dim
            )
        else:
            assign_pred_input_dim = assign_dim
        self.assignPredLayers.append(nn.Linear(assign_pred_input_dim, assign_dim))

        self.init_weights()

    def forward(self, h, adj):
        # equation 5 (GNN_l,pool)
        self.assign_tensor = self.assignLayers(h, adj)
        # equation 6
        self.assign_tensor = nn.Softmax(dim=-1)(
            self.assign_pred_modules(self.assign_tensor)
        )
        # update pooled feature and adj max
        h = torch.matmul(torch.transpose(self.assign_tensor, 0, 1), h)
        adj = (
            torch.transpose(self.assign_tensor, 0, 1)
            @ adj.to_dense()
            @ self.assign_tensor
        )
        ### caculate loss ###
        
        #for loss_layer in self.reg_loss:
        #    loss_name = str(type(loss_layer).__name__)
        #    self.loss_log[loss_name] = loss_layer(adj, anext, s_l)
        #if log:
        #    self.log["a"] = anext.cpu().numpy()
        return h, adj

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, GCNConv):
                m.weight.data = init.xavier_uniform(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)
"""

class DiffPoolReadout(nn.Module):
    def __init__(self):
        super(DiffPoolReadout, self).__init__()

    def forward(self, hs, graph_indicators):
        layer = len(graph_indicators)
        batch = int(max(graph_indicators[0]) + 1)
        h_mean, h_sum, h_max = [], [], []  # torch.zeros_like(batch,layer*dim)
        for i in range(layer):
            h_mean.append(
                scatter(
                    hs[i],
                    graph_indicators[layer - i - 1],
                    dim=0,
                    dim_size=batch,
                    reduce="mean",
                )
            )
            h_sum.append(
                scatter(
                    hs[i],
                    graph_indicators[layer - i - 1],
                    dim=0,
                    dim_size=batch,
                    reduce="add",
                )
            )
            h_max.append(
                scatter(
                    hs[i],
                    graph_indicators[layer - i - 1],
                    dim=0,
                    dim_size=batch,
                    reduce="max",
                )
            )
        readout = torch.cat(
            (
                torch.cat(h_max, dim=1),
                torch.cat(h_sum, dim=1),
                torch.cat(h_mean, dim=1),
            ),
            dim=1,
        )
        return readout
