from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import degree
from ogb.graphproppred import PygGraphPropPredDataset

class DatasetLoader():
    def load_dataset(dataset_name):
        if dataset_name == 'IMDB-MULTI' or dataset_name == 'REDDIT-MULTI-12K':
            dataset = TUDataset('data/', name=dataset_name)
            # transform to one hot 
            max_degree = 0
            for g in dataset:
                if g.edge_index.size(1) > 0:
                    max_degree = max(max_degree, int(degree(g.edge_index[0]).max().item()))
            dataset.transform = OneHotDegree(max_degree)
            
            #args.num_classes = dataset.num_classes
            #args.num_features = dataset.num_features
        elif dataset_name == 'PROTEINS' or dataset_name == 'ENZYMES' or dataset_name == 'DD': 
            dataset = TUDataset('data/', name=dataset_name, use_node_attr=True)
            #args.num_classes = dataset.num_classes
            #args.num_features = dataset.num_features
        elif dataset_name == "ogbg-molhiv":
            dataset = PygGraphPropPredDataset(name = dataset_name, root = 'data/')
        elif dataset_name =="ogbg-code2":
            dataset = PygGraphPropPredDataset(name = dataset_name, root = 'data/') 
        return dataset