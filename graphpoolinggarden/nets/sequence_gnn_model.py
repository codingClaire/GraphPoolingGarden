import torch
from layers.GNN_node import GnnLayer
from layers.GNN_virtual_node import GnnLayerwithVirtualNode
from pooling.poolinglayer import PoolingLayer
from layers.encoders import FeatureEncoder


class sequenceModel(torch.nn.Module):
    def __init__(self, params):
        super(sequenceModel, self).__init__()
        self.emb_dim = params["emb_dim"]
        self.num_tasks = params["num_tasks"]
        self.gnn_type = params["gnn_type"]
        self.num_layer =  params["num_layer"]
        self.graph_pooling = params["pooling"]
        self.emb_dim = params["emb_dim"]
        self.in_dim = params["in_dim"]
        self.dataset_name = params["dataset_name"]
        self.virtual_node = params["virtual_node"]
        # check validation
        if self.gnn_type not in ["gcn", "gin"]:
            raise ValueError("Invalid GNN type.")
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        ### 0. Feature Encode ### 
        self.feature_encoder = FeatureEncoder(self.dataset_name ,self.in_dim,self.emb_dim)
        
        ### 1.GNN to generate node embeddings ###
        if self.virtual_node == "True":
            self.gnn_layer = GnnLayerwithVirtualNode(params)

        else:
            self.gnn_layer = GnnLayer(params)

        ### 2.Pooling method to generate pooling of graph ###
        self.pool_layer = PoolingLayer(params)

        ### 3.Prediction ###
        if self.graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        """gnn_model: input mini-batch graph data and output the prediction result
        """  
        input_feature = self.feature_encoder(batched_data.x)
        
        if self.virtual_node == "True":
            h_node = self.gnn_layer(input_feature,batched_data.edge_index,batched_data.edge_attr,batched_data.batch)
        else:
            h_node = self.gnn_layer(input_feature,batched_data.edge_index,batched_data.edge_attr)
        h_graph = self.pool_layer(h_node, batched_data)

        return self.graph_pred_linear(h_graph)
