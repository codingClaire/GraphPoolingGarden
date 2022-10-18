import torch
import torch.nn.functional as F
from layers.gin_layer import GINConv
from layers.gcn_layer import GCNConv

### GNN to generate node embedding
class SingleGnnLayer(torch.nn.Module):
    """A single gnn layer
    Output:
        node representations
    """
    def __init__(self, params):
        super(SingleGnnLayer, self).__init__()
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


        # single layer of gnn
        if self.gnn_type == 'gin':
            self.gnn_layer = GINConv(self.dataset_name, self.edge_dim,self.emb_dim)
        elif self.gnn_type == 'gcn':
            self.gnn_layer = GCNConv(self.dataset_name, self.edge_dim,self.emb_dim)
        else:
            raise ValueError('Undefined GNN type called {}'.format(self.gnn_type))

        self.batch_norm_layer = torch.nn.BatchNorm1d(self.emb_dim)

    def forward(self, input_feature, edge_index,edge_attr):
        ### computing input node embedding
        h_list = [input_feature]
        h = self.gnn_layer(h_list[0], edge_index, edge_attr)
        h = self.batch_norm_layer(h)
        h = F.dropout(h, self.drop_ratio, training = self.training)
        h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(len(h_list)):
                node_representation += h_list[layer]

        return node_representation
