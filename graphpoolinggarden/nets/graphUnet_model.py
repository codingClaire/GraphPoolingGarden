import torch.nn as nn 
import torch

from layers.encoders import FeatureEncoder, ASTFeatureEncoder
from layers.GNN_node import GnnLayer
from layers.GNN_virtual_node import GnnLayerwithVirtualNode
from layers.single_GNN_node import SingleGnnLayer

from pooling.readoutlayer import ReadoutLayer
from pooling.poolinglayer import PoolingLayer
from unpooling.unpoolinglayer import UnPoolingLayer



class graphUnetModel(nn.Module):
    def __init__(self, params):
        super(graphUnetModel, self).__init__()
        self.emb_dim = params["emb_dim"]
        self.num_tasks = params["num_tasks"]
        self.gnn_type = params["gnn_type"]
        self.num_layer = params["num_layer"]
        self.graph_pooling = params["pooling"]
        self.pool_num_layer = params["pool_num"]
        self.emb_dim = params["emb_dim"]
        self.in_dim = params["in_dim"]  # original node in_dim
        self.dataset_name = params["dataset_name"]
        self.virtual_node = params["virtual_node"]
        self.ks = params["ks"]

        # check validation
        if self.gnn_type not in ["gcn", "gin"]:
            raise ValueError("Invalid GNN type.")
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        #### feature encoder part ####
        if self.dataset_name == "ogbg-code2":
            # special for ogbg-code2
            self.max_seq_len = params["max_seq_len"]
            self.num_vocab = params["num_vocab"] + 2  # for <unk> and <eos>
            self.feature_encoder = ASTFeatureEncoder(
                self.emb_dim,
                params["num_nodetypes"],
                params["num_nodeattributes"],
                params["max_depth"],
            )
        else:
            self.feature_encoder = FeatureEncoder(
                self.dataset_name, self.in_dim, self.emb_dim
            )
        self.inputGnnLayer = SingleGnnLayer(params)
        ##### gnn part #####
        self.upGnnLayers = nn.ModuleList()
        self.downGnnLayers = nn.ModuleList()
        self.bottomGnnLayer = SingleGnnLayer(params)

        ##### pooling part ####
        self.poolLayers = nn.ModuleList()
        self.unpoolLayers = nn.ModuleList()

        for i in range(self.pool_num_layer):
            ### 1.GNN to generate node embeddings ###
            if self.virtual_node == "True":
                self.upGnnLayers.append(GnnLayerwithVirtualNode(params))
                self.downGnnLayers.append(GnnLayerwithVirtualNode(params))
            else:
                self.upGnnLayers.append(GnnLayer(params))
                self.downGnnLayers.append(GnnLayer(params))
            
            ### 2.Pooling method to generate pooling of graph ###
            self.poolLayers.append(PoolingLayer(params,k=self.ks[i]))
            self.unpoolLayers.append(UnPoolingLayer(params))
        
        ### 3. readout ### 
        self.readoutLayer = ReadoutLayer(params)

        ### 4.Prediction ###
        self.graph_pred_linear_list = nn.ModuleList()
        if self.graph_pooling == "set2set":
            if self.dataset_name == "ogbg-code2":
                for _ in range(self.max_seq_len):
                    self.graph_pred_linear_list.append(
                        nn.Linear(2 * self.emb_dim, self.num_vocab)
                    )
            else:
                self.graph_pred_linear = nn.Linear(2 * self.emb_dim, self.num_tasks)
        else:
            if self.dataset_name == "ogbg-code2":
                for _ in range(self.max_seq_len):
                    self.graph_pred_linear_list.append(
                        nn.Linear(3*self.emb_dim*(self.pool_num_layer+1), self.num_vocab)
                    )
            else:
                self.graph_pred_linear = nn.Linear(3*self.emb_dim*(self.pool_num_layer+1), self.num_tasks)

    def forward(self, batched_data):
        edge_index = batched_data.edge_index
        batch = batched_data.batch
        x = batched_data.x
       
        edge_attr = batched_data.edge_attr

        ## 1. encode input feature
        if self.dataset_name == "ogbg-code2":
            input_feature = self.feature_encoder(
                x,
                batched_data.node_depth.view(
                    -1,
                ),
            )
        else:
            input_feature = self.feature_encoder(x)
        
        ## 2. Graph U-net like model
        ori_h = self.inputGnnLayer(input_feature,edge_index, edge_attr)
        ori_batch = batch
        h = ori_h
        edge_index_list, edge_attr_list = [],[]
        graph_indicator_list = []
        graph_indicator_list.append(ori_batch)
        down_outs = []
        hs = []
        indices_list = [] # the idx preserved after pooling layer 
        # down layers
        edge_attr = None
        for i in range(self.pool_num_layer):
            if self.virtual_node == "True":
                h = self.downGnnLayers[i](h, edge_index, edge_attr, batch)
            else:
                h = self.downGnnLayers[i](h, edge_index, edge_attr)
            edge_index_list.append(edge_index)
            edge_attr_list.append(edge_attr)
            graph_indicator_list.append(batch)
            down_outs.append(h)
            h,edge_index,edge_attr,batch,idxs= self.poolLayers[i](h, edge_index,edge_attr, batch)
            # save for up-operation
            indices_list.append(idxs)
            
        # bottom layer
        h = self.bottomGnnLayer(h, edge_index, edge_attr)
        # up layers
        for i in range(self.pool_num_layer):
            up_idx = self.pool_num_layer - i -1
            edge_index,edge_attr = edge_index_list[up_idx], edge_attr_list[up_idx]
            idxs, batch = indices_list[up_idx], graph_indicator_list[up_idx+1]
            h = self.unpoolLayers[i](batch.shape[0],h,idxs)

            if self.virtual_node == "True":
                h = self.upGnnLayers[i](h, edge_index, edge_attr,batch)
            else:
                h = self.upGnnLayers[i](h, edge_index, edge_attr)
            
            h = h.add(down_outs[up_idx])
            hs.append(h)
        h = h.add(ori_h)
        hs.append(h)
        
        ## 3. multiple readout layer
        graph_representation = self.readoutLayer(hs, graph_indicator_list)
        
        ## 4. prediction
        if self.dataset_name == "ogbg-code2":
            pred_list = []
            for i in range(self.max_seq_len):
                pred_list.append(self.graph_pred_linear_list[i](graph_representation))
            return pred_list
        else:
            return self.graph_pred_linear(graph_representation)

    def get_model_loss(self):
        return 0
