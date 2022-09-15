import torch
import torch.nn.functional as F
from layers.gin_layer import GINConv
from layers.gcn_layer import GCNConv
from layers.encoders import FeatureEncoder

### GNN to generate node embedding
class SingleGnnLayer(torch.nn.Module):
    """A single gnn layer
    Output:
        node representations
    """
    def __init__(self, params):
        super(SingleGnnLayer, self).__init__()
        self.num_layer =  params["num_layer"]
        self.drop_ratio = params["drop_ratio"]
        self.JK = params["JK"]
        self.residual = params["residual"]
        self.gnn_type = params["gnn_type"]
        self.emb_dim = params["emb_dim"]
        self.in_dim = params["in_dim"]
        self.dataset_name = params["dataset_name"]

        self.feature_encoder = FeatureEncoder(self.dataset_name ,self.in_dim,self.emb_dim)
        # single layer of gnn
        if self.gnn_type == 'gin':
            self.gnn_layer = GINConv(self.dataset_name, self.in_dim,self.emb_dim)
        elif self.gnn_type == 'gcn':
            self.gnn_layer = GCNConv(self.dataset_name, self.in_dim,self.emb_dim)
        else:
            raise ValueError('Undefined GNN type called {}'.format(self.gnn_type))

        self.batch_norm_layer = torch.nn.BatchNorm1d(self.emb_dim)

    def forward(self, batched_data):
        x, edge_index, edge_attr = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
        )
    
        ### computing input node embedding
        h_list = [self.feature_encoder(x)]
        #calculate h
        h = self.gnn_layer(h_list[0], edge_index, edge_attr)
        h = self.batch_norm_layer(h)
        h = F.dropout(h, self.drop_ratio, training = self.training)
        if self.residual:
            h += h_list[layer]
        # append h
        h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(len(h_list)):
                node_representation += h_list[layer]

        return node_representation
