import torch
import torch.nn.functional as F
from layers.gin_layer import GINConv
from layers.gcn_layer import GCNConv

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
        self.in_dim = params["in_dim"]
        self.dataset_name = params["dataset_name"]
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for _ in range(self.num_layer):
            if self.gnn_type == 'gin':
                self.convs.append(GINConv(self.dataset_name, self.in_dim,self.emb_dim))
            elif self.gnn_type == 'gcn':
                self.convs.append(GCNConv(self.dataset_name, self.in_dim,self.emb_dim))
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
