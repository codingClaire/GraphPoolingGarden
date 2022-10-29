import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool

from layers.encoders import FeatureEncoder, ASTFeatureEncoder
from layers.GNN_node import GnnLayerwithAdj
from pooling.diffpooling_layer import DiffPool
from pooling.readoutlayer import ReadoutLayer


def batch2tensor(batch_adj, batch_feat, node_per_pool_graph):
    """
    code snippet contributed by dgl
    transform a batched graph to batched adjacency tensor and node feature tensor
    the basic idea is that the hyper-nodes generated after pooling will be seperated to each graph on average
    """
    batch_size = int(batch_adj.size()[0] / node_per_pool_graph)
    adj_list = []
    feat_list = []
    for i in range(batch_size):
        start = i * node_per_pool_graph
        end = (i + 1) * node_per_pool_graph
        adj_list.append(batch_adj[start:end, start:end])
        feat_list.append(batch_feat[start:end, :])
    adj_list = list(map(lambda x: torch.unsqueeze(x, 0), adj_list))
    feat_list = list(map(lambda x: torch.unsqueeze(x, 0), feat_list))
    adj = torch.cat(adj_list, dim=0)
    feat = torch.cat(feat_list, dim=0)

    return feat, adj


class diffPoolModel(nn.Module):

    def __init__(self, params):
        super(diffPoolModel, self).__init__()
        self.emb_dim = params["emb_dim"]
        self.num_tasks = params["num_tasks"]
        self.gnn_type = params["gnn_type"]
        self.num_layer = params["num_layer"]
        self.graph_pooling = params["pooling"]
        self.pool_num_layer = params["pool_num"]
        self.emb_dim = params["emb_dim"]
        self.in_dim = params["in_dim"]  # original node in_dim
        self.dataset_name = params["dataset_name"]

        # special for diffpool
        self.assign_ratio = params["diffpool"]["assign_ratio"]
        self.max_num_nodes = params["diffpool"]["max_num_nodes"]
        self.assign_dim = int(self.max_num_nodes * self.assign_ratio)
        self.num_aggs =  params["diffpool"]["num_aggs"]

        # check validation
        if self.gnn_type not in ["gcn", "gin"]:
            raise ValueError("Invalid GNN type.")
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        #### feature encoder pabrt ####
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
        self.inputGnnLayer = GnnLayerwithAdj(params, two_dim=True, in_dim=self.emb_dim)
        # based on the output of the inputGnnLayer
        if params["JK"] == "concat":
            self.pool_in_dim = self.emb_dim * self.num_layer
        else:
            self.pool_in_dim = self.emb_dim
        self.firstDiffpoolLayer = DiffPool(params["diffpool"], self.pool_in_dim, self.assign_dim, two_dim= True)
        self.firstGnnLayer = GnnLayerwithAdj(params, in_dim=self.emb_dim)
        self.assign_dim = int(self.assign_dim * self.assign_ratio)
        self.gnnLayers = nn.ModuleList()
        self.diffpoolLayers = nn.ModuleList()
        for _ in range(self.pool_num_layer - 1):
            self.gnnLayers.append(GnnLayerwithAdj(params,two_dim=False))
            self.diffpoolLayers.append(
                DiffPool(params["diffpool"], self.pool_in_dim, self.assign_dim,two_dim=False)
            )
            self.assign_dim = int(self.assign_dim * self.assign_ratio)

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
                        nn.Linear(
                            3 * self.emb_dim * (self.pool_num_layer + 1), self.num_vocab
                        )
                    )
            else:
                self.graph_pred_linear = nn.Linear(
                    3 * self.emb_dim * (self.pool_num_layer + 1), self.num_tasks
                )

    def forward(self, batched_data):
        edge_index = batched_data.edge_index
        edge_attr = batched_data.edge_attr
        batch = batched_data.batch
        x = batched_data.x
        node_num = x.shape[0]

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
        ## 1.5 convert to adj
        edge_attr = torch.ones([edge_index.shape[1]]).to(batch.device)
        adj = torch.sparse.FloatTensor(
            edge_index, edge_attr, torch.Size([node_num, node_num])
        )
        adj = adj.to_dense()

        ## 2. Diffpool like model
        # Actually, it is quite similar to the hierarchical_gnn_model but with slight differences
        out_alls, graph_indicators = [], []
        embedding_tensor = self.inputGnnLayer(input_feature,adj)
        out = global_max_pool(embedding_tensor, batch)
        out_alls.append(out)
        if self.num_aggs == 2:
            out = global_add_pool(embedding_tensor, batch)
            out_alls.append(out)
        # first layer
        h, adj = self.firstDiffpoolLayer(embedding_tensor,adj)
        node_per_pool_graph = int(adj.shape[0] / (batch[-1]+1))
        h, adj = batch2tensor(adj, h, node_per_pool_graph)
        h = self.firstGnnLayer(h, adj)
        out = torch.sum(h, dim=1)
        out_alls.append(out)

        for i in range(self.pool_num_layer):
            h, adj = self.diffpoolLayers[i](h, adj)
            h = self.gnnLayers[i](h, adj)
            # TODO: debug from here!! 
            out = global_max_pool(h, batch)
            out_alls.append(out)
            if self.num_aggs == 2:
                out = global_add_pool(h, batch)
                out_alls.append(out)

        ## 3. multiple readout layer
        graph_representation = self.readoutLayer(out_alls, graph_indicators)

        ## 4. prediction
        if self.dataset_name == "ogbg-code2":
            pred_list = []
            for i in range(self.max_seq_len):
                pred_list.append(self.graph_pred_linear_list[i](graph_representation))
            return pred_list
        else:
            return self.graph_pred_linear(graph_representation)

    def total_loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        for _, value in self.firstPoolLayer.loss_log.items():
            loss += value
        for layer in self.diffpoolLayers:
            for _, value in layer.loss_log.items():
                loss += value
        return loss
