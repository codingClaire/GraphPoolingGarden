import torch
import torch.nn as nn
from torch.nn import init
from torch_geometric.nn import global_mean_pool, global_max_pool

from layers.gcn_layer import GCNConv
from layers.GNN_node import GnnLayer
from layers.GNN_virtual_node import GnnLayerwithVirtualNode


class DiffPoolReadout(torch.nn.Module):
    def __init__(self, params):
        super(diffPoolReadout, self).__init__()
        self.diffpool_params = params["diffpool_params"]

class diffPoolReadout(torch.nn.Module):
    def __init__(self, params, embed_dim):
        super(diffPoolReadout, self).__init__()
        self.diffpool_params = params["diffpool_params"]
        self.embed_dim = embed_dim

        self.pools = DiffPool(self.diffpool_params, embed_dim)
        self.linear = torch.nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, input_feature, batched_data):
        """input the graph information and output the readout result"""
        edge_index = batched_data.edge_index
        edge_attr = batched_data.edge_attr
        graph_indicator = batched_data.batch
        input_feature, _, _, graph_indicator = self.pools(
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



class DiffPool(nn.Module):
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, GCNConv):
                m.weight.data = init.xavier_uniform(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)
    

    def __init__(self, params, embed_dim):
        super(DiffPool, self).__init__()
        self.max_num_nodes = params["max_num_nodes"]
        self.pool_num = params["pool_num"]
        self.assign_hidden_dim = params["assign_hidden_dim"]
        self.assign_ratio = params["assign_ratio"]
        self.assign_num_layers = params["assign_num_layers"]
        self.concat = params["concat"]
        self.bias = params["bias"]
        self.add_self = not self.concat
        if params["activation"] == "ReLU":
            self.activation = nn.ReLU()

        self.hidden_dim = params["assign_hidden_dim"]
        if self.concat:
            self.pred_input_dim = self.hidden_dim * (self.assign_num_layers - 1) + embed_dim
        else:
            self.pred_input_dim = embed_dim



        # GCN(GNN_l,pool)
        gnnLayers = torch.nn.ModuleList()
        for _ in range(self.pool_layer_num):
            ### 1.GNN to generate node embeddings ###
            if self.virtual_node == "True":
                gnnLayers.append(GnnLayerwithVirtualNode(params))
            else:
                gnnLayers.append(GnnLayer(params))
        
        # assignment(GNN_l,pool)
        self.assignLayers = torch.nn.ModuleList()
        self.assignPredLayers = torch.nn.ModuleList()
        assign_dim = int(self.max_num_nodes * self.assign_ratio)
        for _ in range(self.pool_num):
            #### assignLayers #### 
            assign_params = params
            assign_params["in_dim"] = self.pred_input_dim
            assign_params["emb_dim"] = assign_dim
            self.assignLayers.append(GnnLayer(assign_params))
            #### assignPredLayers #### 
            if self.concat:
                assign_pred_input_dim = (
                    self.assign_hidden_dim * (self.assign_num_layers - 1) + assign_dim
                )
            else:
                assign_pred_input_dim= assign_dim
            self.assignPredLayers.append(nn.Linear(assign_pred_input_dim, assign_dim))
            assign_dim = int(assign_dim * self.assign_ratio)
    
        self.init_weights()
    
    def forward(self, input_feature, batched_data):
        edge_index = batched_data.edge_index
        edge_attr = batched_data.edge_attr
        graph_indicator = batched_data.batch

        assign_feature = input_feature
        # equation 5
        embedding_tensor = self.self.gnnLayers[0](
            assign_feature, edge_index, edge_attr
        )                
        for i in range(self.pool_num):
            # equation 5 (GNN_l,pool)
            self.assign_tensor = self.assignLayers[i](
                input_feature, edge_index, edge_attr
            )
            # equation 6
            self.assign_tensor = nn.Softmax(dim=-1)(
                self.assign_pred_modules[i](self.assign_tensor)
            )
            # equation 3 
            input_feature = torch.matmul(
                torch.transpose(self.assign_tensor, 0, 1), embedding_tensor
            )
            assign_feature = input_feature

            # TODO :update adj --> edge_attr --> edge_index
            adj = (
                torch.transpose(self.assign_tensor, 0, 1)
                @ adj.to_dense()
                @ self.assign_tensor
            )
            
            #edge_index, edge_attr = filter_adj(
            #edge_index, edge_attr, mask, num_nodes=attn_score.size(0)


            # TODO: update graph_indicator

            embedding_tensor = self.gnnLayers[i + 1](
                input_feature, edge_index, edge_attr
            )
        return embedding_tensor, graph_indicator