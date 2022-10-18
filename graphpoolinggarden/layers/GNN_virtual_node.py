import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from layers.gin_layer import GINConv
from layers.gcn_layer import GCNConv

### Virtual GNN to generate node embedding
class GnnLayerwithVirtualNode(torch.nn.Module):
    def __init__(self, params):
        super(GnnLayerwithVirtualNode, self).__init__()
        self.num_layer =  params["num_layer"]
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
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, self.emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for _ in range(self.num_layer):
            if self.gnn_type == "gin":
                self.convs.append(GINConv(self.dataset_name, self.edge_dim, self.emb_dim))
            elif self.gnn_type == "gcn":
                self.convs.append(GCNConv(self.dataset_name, self.edge_dim,self.emb_dim))
            else:
                raise ValueError("Undefined GNN type called {}".format(self.gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(self.emb_dim))

        for _ in range(self.num_layer - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.emb_dim, 2 * self.emb_dim),
                    torch.nn.BatchNorm1d(2 * self.emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * self.emb_dim, self.emb_dim),
                    torch.nn.BatchNorm1d(self.emb_dim),
                    torch.nn.ReLU(),
                )
            )

    def forward(self, input_feature, edge_index,edge_attr,batch):

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
        )
        h_list = [input_feature]
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = (
                    global_add_pool(h_list[layer], batch) + virtualnode_embedding
                )
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                        self.drop_ratio,
                        training=self.training,
                    )
                else:
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                        self.drop_ratio,
                        training=self.training,
                    )

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation
