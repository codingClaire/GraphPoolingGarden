import torch
import torch.nn as nn
from torch.nn import init

from layers.gcn_layer import GCNConv
from layers.GNN_node import GnnLayerwithAdj


class EntropyLoss(nn.Module):
    # Return Scalar
    def forward(self, adj, assign_tensor):
        entropy = (
            (torch.distributions.Categorical(probs=assign_tensor).entropy())
            .sum(-1)
            .mean(-1)
        )
        assert not torch.isnan(entropy)
        return entropy


class LinkPredLoss(nn.Module):
    def forward(self, adj, assign_tensor):
        if (len(adj.shape)) == 2:
            link_pred_loss = torch.norm(
                adj - assign_tensor.matmul(assign_tensor.transpose(-1, -2))
            )
            link_pred_loss = link_pred_loss / (adj.size(0) * adj.size(1))
            return link_pred_loss
        else:
            link_pred_loss = (
                adj - assign_tensor.matmul(assign_tensor.transpose(-1, -2))
            ).norm(dim=(1, 2))
            link_pred_loss = link_pred_loss / (adj.size(1) * adj.size(2))
            return link_pred_loss.mean()


class DiffPool(nn.Module):
    def __init__(self, params, in_dim, assign_dim, **kwargs):
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
        assign_params["device"] = kwargs["device"]
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
        self.assignLayers = GnnLayerwithAdj(
            assign_params, two_dim=self.two_dim, in_dim=in_dim
        )
        self.assignPredLayer = nn.Linear(assign_pred_input_dim, assign_dim)

        # loss
        self.pooling_loss = {}
        self.lossLayers = nn.ModuleList([])
        if assign_params["link_pred"] == "True":
            self.lossLayers.append(LinkPredLoss())
        if assign_params["entropy"] == "True":
            self.lossLayers.append(EntropyLoss())

        self.init_weights()

    def forward(self, h, adj):
        # equation 5 (GNN_l,pool)
        assign_tensor = self.assignLayers(h, adj)
        # equation 6
        assign_tensor = self.assignPredLayer(assign_tensor)
        assign_tensor = nn.Softmax(dim=-1)(assign_tensor)
        # update pooled feature and adj max
        h = torch.matmul(torch.transpose(assign_tensor, -1, -2), h)
        adj_next = torch.transpose(assign_tensor, -1, -2) @ adj @ assign_tensor
        # calculate the loss
        for lossLayer in self.lossLayers:
            loss_name = str(type(lossLayer).__name__)
            self.pooling_loss[loss_name] = lossLayer(adj, assign_tensor)
        return h, adj_next

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, GCNConv):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)
