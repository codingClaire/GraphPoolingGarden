# GraphPoolingGarden
A repo for baseline of graph pooling methods.

[中文](/README_CN.md)

## Datasets
- TuDataset
  - D&D
  - PROTEINS
  - ENZYMES
  - NCI1/NCI109
  - Reddit-Binary
- OGB
  - ogbg-molhiv
  - ogbg-ppa
  - ogbg-code2

## Pooling methods
- set2set
- sagpool(sequence/hierarchical)
- graph-U-net
- Diffpool

## Model Framework

- hierarchical model
- sequencial model
- U-net-like model: only for graph U-net model
- diffpooling model: only for diffpool model 

## Readout methods
- mean
- max
- sum

## ConvLayer

- GCN
- GIN


## Usage

### 1. create the config json file
create a `config.json` in `configs` folder, an example is like this: 

```json
{
    "dataset_name": ["ENZYMES"],
    "batch_size": 2,
    "epochs": 100,
    "seed": [1,2,3,4,5],
    "model":"global",
    "gnn_type": "gcn",
    "num_layer": 3,
    "emb_dim": 300,
    "drop_ratio": 0.5,
    "virtual_node": "False",
    "residual": "False",
    "JK": "last",
    "pooling": "sagpool",
    "sagpool": {
        "keep_ratio": 0.8,
        "activation": "tanh",
        "layer_num": 1
    }
}
```

When the key is a list such as dataset_name, all permutation will be trained and log in the csv file named as the comabination of the key values of 'dataset_name' and 'pooling'. In the examle, the final result will be saved in ENZYMES_sagpool.csv.

Warn: 

Not all values can be written list-like, only the following keys can:
- dataset_name
- seed

### 2. running train.py to train, valid and test

`python graphpoolinggarden/train.py --config configs/config.json`


## Acknowledgement

The program is built based on the code of the following released codes and programs:
* [Open Graph Benchmark(OGB)](https://github.com/snap-stanford/ogb)
* [Graph-U-Nets](https://github.com/HongyangGao/Graph-U-Nets)
* [DiffPool](https://github.com/RexYing/diffpool)
* [DGL](https://github.com/dmlc/dgl)

We appreciate the authors' effort for the contribution to the research community.

