import torch
from layers.GNN_node import GnnLayer
from layers.GNN_virtual_node import GnnLayerwithVirtualNode
from pooling.poolinglayer import PoolingLayer
from layers.encoders import FeatureEncoder

class hierarchicalModel(torch.nn.Module):
    def __init__(self, params):
        super(hierarchicalModel, self).__init__()
        self.emb_dim = params["emb_dim"]
        self.num_tasks = params["num_tasks"]
        self.gnn_type = params["gnn_type"]
        self.num_layer = params["num_layer"]
        self.graph_pooling = params["pooling"]
        self.pool_num = params["pool_num"]
        self.emb_dim = params["emb_dim"]
        self.in_dim = params["in_dim"]
        self.dataset_name = params["dataset_name"]
        self.virtual_node = params["virtual_node"]
        # check validation
        if self.gnn_type not in ["gcn", "gin"]:
            raise ValueError("Invalid GNN type.")
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.feature_encoder = FeatureEncoder(self.dataset_name ,self.in_dim,self.emb_dim)
        self.gnnLayers = torch.nn.ModuleList()
        self.poolLayers = torch.nn.ModuleList()
        for _ in range(self.pool_num):
            ### 1.GNN to generate node embeddings ###
            if self.virtual_node == "True":
                self.gnnLayers.append(GnnLayerwithVirtualNode(params))
            else:
                self.gnnLayers.append(GnnLayer(params))
            ### 2.Pooling method to generate pooling of graph ###
            self.poolLayers.append(PoolingLayer(params))

        ### 3.Prediction ###
        if self.graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        input_feature = self.feature_encoder(batched_data.x)
        readoutList =[]
        for i in range(self.pool_num):
            if self.virtual_node == "True":
                 embedding_tensor= self.gnnLayers[i](
                    input_feature,
                    batched_data.edge_index,
                    batched_data.edge_attr,
                    batched_data.batch,
                )
            else:
                 embedding_tensor = self.gnnLayers[i](
                    input_feature, 
                    batched_data.edge_index, 
                    batched_data.edge_attr
                )
            readout  = self.poolLayers[i](embedding_tensor, batched_data)
            readoutList.append(readout)
        node_representation = 0
        for layer in range(len(readoutList)):
            node_representation += readoutList[layer]
        return self.graph_pred_linear(node_representation)
