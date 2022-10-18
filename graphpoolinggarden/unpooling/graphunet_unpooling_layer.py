import torch.nn as nn

class GraphUnetUnpooling(nn.Module):
    def __init__(self):
        super(GraphUnetUnpooling, self).__init__()

    def forward(self, node_num, h, idxs):
        new_h = h.new_zeros([node_num, h.shape[1]])
        new_h[idxs] = h
        return new_h