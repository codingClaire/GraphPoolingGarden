import torch
from layers.GNN_node import GnnLayer
from layers.GNN_virtual_node import GnnLayerwithVirtualNode
from pooling.poolinglayer import PoolingLayer
from layers.encoders import FeatureEncoder,ASTFeatureEncoder

class hierarchicalModel(torch.nn.Module):
    def __init__(self, params):
        super(hierarchicalModel, self).__init__()
        self.emb_dim = params["emb_dim"]
        self.num_tasks = params["num_tasks"]
        self.gnn_type = params["gnn_type"]
        self.num_layer = params["num_layer"]
        self.graph_pooling = params["pooling"]
        self.pool_num_layer = params["pool_num"]
        self.emb_dim = params["emb_dim"]
        self.in_dim = params["in_dim"] # original node in_dim
        self.dataset_name = params["dataset_name"]
        self.virtual_node = params["virtual_node"]
        self.max_seq_len = params["max_seq_len"]
        self.num_vocab = params["num_vocab"] + 2 # for <unk> and <eos>

        # check validation
        if self.gnn_type not in ["gcn", "gin"]:
            raise ValueError("Invalid GNN type.")
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        if self.dataset_name == "ogbg-code2":
            #### special for ogbg-code2 #### 
            self.feature_encoder = ASTFeatureEncoder(self.emb_dim,
            params["num_nodetypes"],params["num_nodeattributes"], params["max_depth"])
        else:
            self.feature_encoder = FeatureEncoder(self.dataset_name ,self.in_dim,self.emb_dim)
        self.gnnLayers = torch.nn.ModuleList()
        self.poolLayers = torch.nn.ModuleList()
        for _ in range(self.pool_num_layer):
            ### 1.GNN to generate node embeddings ###
            if self.virtual_node == "True":
                self.gnnLayers.append(GnnLayerwithVirtualNode(params))
            else:
                self.gnnLayers.append(GnnLayer(params))
            ### 2.Pooling method to generate pooling of graph ###
            self.poolLayers.append(PoolingLayer(params))

        ### 3.Prediction ###
        self.graph_pred_linear_list = torch.nn.ModuleList()
        if self.graph_pooling == "set2set":
            if self.dataset_name =="ogbg-code2":
                for _ in range(self.max_seq_len):
                    self.graph_pred_linear_list.append(
                        torch.nn.Linear(2* self.emb_dim, self.num_vocab)
                    )
            else:
                self.graph_pred_linear = torch.nn.Linear(2 * self.emb_dim, self.num_tasks)
        else:
            if self.dataset_name =="ogbg-code2":
                for _ in range(self.max_seq_len):
                    self.graph_pred_linear_list.append(
                        torch.nn.Linear(self.emb_dim, self.num_vocab)
                    )
            else:
                self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)


    def forward(self, batched_data):
        ## 1. encode input feature
        if(self.dataset_name ==  "ogbg-code2"):
            input_feature = self.feature_encoder(batched_data.x, batched_data.node_depth.view(-1,))
        else: 
            input_feature = self.feature_encoder(batched_data.x)
        ## 2. (gnn layer * num_layer times + pool layer) * pool_num_layer times
        readoutList =[]
        for i in range(self.pool_num_layer):
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
        ## 3. multiple readout layer 
        graph_representation = 0
        for layer in range(len(readoutList)):
            graph_representation += readoutList[layer]
        ## 4. prediction 
        if(self.dataset_name ==  "ogbg-code2"):
            pred_list = []
            for i in range(self.max_seq_len):
                pred_list.append(self.graph_pred_linear_list[i](graph_representation))
            return pred_list
        else:
            return self.graph_pred_linear(graph_representation)