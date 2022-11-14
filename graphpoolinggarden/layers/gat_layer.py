from torch_geometric.nn import MessagePassing
from layers.encoders import EdgeEncoder
from torch_geometric.nn import GATConv


class GraphAttentionConv(MessagePassing):
    def __init__(self, dataset_name, edge_dim,emb_dim):
        super(GraphAttentionConv, self).__init__()
        self.edge_encoder = EdgeEncoder(dataset_name, edge_dim,emb_dim)
        self.gatconv = GATConv(emb_dim, emb_dim,edge_dim = emb_dim)

    def forward(self,x,edge_index,edge_attr):
        if edge_attr != None and len(edge_attr.shape) != 1:
            edge_emb = self.edge_encoder(edge_attr)
            x = self.gatconv(x,edge_index,edge_emb)
        else:
            x = self.gatconv(x,edge_index)
        return x