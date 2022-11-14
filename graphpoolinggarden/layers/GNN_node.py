import torch
import torch.nn.functional as F
from torch.nn import init
from layers.gin_layer import GINConv
from layers.gcn_layer import GCNConv
from layers.gcn_layer import GCNConvwithAdj
from layers.graphsage_layer import GraphSAGEConv
from layers.gat_layer import GraphAttentionConv

### GNN to generate node embedding
class GnnLayer(torch.nn.Module):
    def __init__(self, params):
        super(GnnLayer, self).__init__()
        self.num_layer =  params["num_layer"]
        self.drop_ratio = params["drop_ratio"]
        self.JK = params["JK"]
        self.residual = params["residual"]
        self.gnn_type = params["gnn_type"]
        self.emb_dim = params["emb_dim"]
        if "edge_dim" in params.keys():
            # currently only for ogbg-code2 dataset
            self.edge_dim  = params["edge_dim"]
        else:
            # for this situation in conv layer
            # edge_dim is useless
            self.edge_dim  = params["in_dim"]
        self.dataset_name = params["dataset_name"]
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for _ in range(self.num_layer):
            if self.gnn_type == 'gin':
                self.convs.append(GINConv(self.dataset_name, self.edge_dim,self.emb_dim))
            elif self.gnn_type == 'gcn':
                self.convs.append(GCNConv(self.dataset_name,self.edge_dim,self.emb_dim))
            elif self.gnn_type == 'graphsage':
                self.convs.append(GraphSAGEConv(self.dataset_name, self.edge_dim, self.emb_dim))
            elif self.gnn_type == 'gat':
                self.convs.append(GraphAttentionConv(self.dataset_name, self.edge_dim, self.emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(self.gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(self.emb_dim))

    def forward(self, input_feature, edge_index,edge_attr):
        ### computing input node embedding
        h_list = [input_feature]
        for layer in range(self.num_layer):
            # conv --> batchnorm --> relu --> dropout
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation



### GNN to generate node embedding
class GnnLayerwithAdj(torch.nn.Module):
    """Summary
    The class is used for multiple stacked gnnlayers implementation with h and adj as input.
    In DiffPool, the adjacency matrix are not discrete, 
    therefore cannot convert to edge_index and edge_attr and use GnnLayer class
    """
    def __init__(self, params,**kwargs):
        super(GnnLayerwithAdj, self).__init__()
        self.num_layer =  params["num_layer"]
        self.drop_ratio = params["drop_ratio"]
        self.device = params["device"]
        if "JK" not in params.keys():
            self.JK = "last"
        else:
            self.JK = params["JK"]
        if "residual" not in params.keys():
            self.residual = "False"
        else:
            self.residual = params["residual"]
        if "gnn_type" not in params.keys():
            self.gnn_type = "gcn"
        else:
            self.gnn_type = params["gnn_type"]
        self.emb_dim = params["emb_dim"]
        self.two_dim = kwargs["two_dim"] if "two_dim" in kwargs.keys() else False
        self.in_dim = kwargs["in_dim"] if "in_dim" in kwargs.keys() else self.emb_dim
        self.dataset_name = params["dataset_name"]
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        if "tensorized" in kwargs.keys():
            self.tensorized = kwargs["tensorized"]
        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        # self.batch_norms = torch.nn.ModuleList()

        if self.gnn_type == 'gcn':
                self.convs.append(GCNConvwithAdj(self.in_dim,self.emb_dim,self.drop_ratio,self.device))
        else:
            raise ValueError('Undefined GNN type called {}'.format(self.gnn_type))
        for _ in range(self.num_layer-1):
            if self.gnn_type == 'gcn':
                self.convs.append(GCNConvwithAdj(self.emb_dim,self.emb_dim,self.drop_ratio,self.device))
            else:
                raise ValueError('Undefined GNN type called {}'.format(self.gnn_type))
        
            # self.batch_norms.append(torch.nn.BatchNorm1d(self.emb_dim))
        
        self.init_weights()

    def batchNorm(self, x):
        # both are work in 2d and 3d data
        bn_module = torch.nn.BatchNorm1d(x.size()[1]).to(self.device)
        return bn_module(x)


    def forward(self, h, adj):
        ### computing input node embedding
        cat_dim =1 if self.two_dim == True else 2
        h_list = []
        for layer in range(self.num_layer):
            # the dropout operation is in self.convs[layers]
            h = self.convs[layer](h, adj)
            if layer != self.num_layer - 1:
                h = self.batchNorm(F.relu(h))
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        elif self.JK == "concat":
            node_representation = torch.cat(h_list,dim=cat_dim)
        return node_representation

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, GCNConvwithAdj):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=torch.nn.init.calculate_gain("relu")
                )
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)