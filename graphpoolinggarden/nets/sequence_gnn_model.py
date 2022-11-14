import torch
from layers.GNN_node import GnnLayer
from layers.GNN_virtual_node import GnnLayerwithVirtualNode
from pooling.readoutlayer import ReadoutLayer
from layers.encoders import FeatureEncoder, ASTFeatureEncoder


class sequenceModel(torch.nn.Module):
    def __init__(self, params):
        super(sequenceModel, self).__init__()
        self.emb_dim = params["emb_dim"]
        self.num_tasks = params["num_tasks"]
        self.gnn_type = params["gnn_type"]
        self.num_layer = params["num_layer"]
        self.graph_pooling = params["pooling"]
        self.emb_dim = params["emb_dim"]
        self.in_dim = params["in_dim"]
        self.dataset_name = params["dataset_name"]
        self.virtual_node = params["virtual_node"]
        # check validation
        if self.gnn_type not in ["gcn", "gin","graphsage","gat"]:
            raise ValueError("Invalid GNN type.")
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        ### 0. Feature Encode ###
        if self.dataset_name == "ogbg-code2":
            #### special for ogbg-code2 ####
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
        ### 1.GNN to generate node embeddings ###
        if self.virtual_node == "True":
            self.gnn_layer = GnnLayerwithVirtualNode(params)

        else:
            self.gnn_layer = GnnLayer(params)

        ### 2.Pooling and Readout to generate pooling of graph ###
        self.readout_layer = ReadoutLayer(params)

        ### 3.Prediction ###
        self.graph_pred_linear_list = torch.nn.ModuleList()
        if self.graph_pooling == "set2set":
            if self.dataset_name == "ogbg-code2":
                for _ in range(self.max_seq_len):
                    self.graph_pred_linear_list.append(
                        torch.nn.Linear(2 * self.emb_dim, self.num_vocab)
                    )
            else:
                self.graph_pred_linear = torch.nn.Linear(
                    2 * self.emb_dim, self.num_tasks
                )
        else:
            if self.dataset_name == "ogbg-code2":
                for _ in range(self.max_seq_len):
                    self.graph_pred_linear_list.append(
                        torch.nn.Linear(self.emb_dim, self.num_vocab)
                    )
            else:
                self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        """gnn_model: input mini-batch graph data and output the prediction result"""
        edge_index, edge_attr = batched_data.edge_index, batched_data.edge_attr
        x, batch = batched_data.x, batched_data.batch
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
        ## 2. gnn layer * num_layer times
        if self.virtual_node == "True":
            h_node = self.gnn_layer(input_feature, edge_index, edge_attr, batch)
        else:
            h_node = self.gnn_layer(input_feature, edge_index, edge_attr)

        ## 3. pool layer+ readout
        # TODO: addd multiple pool layers situation
        graph_representation = self.readout_layer(h_node, batched_data)

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