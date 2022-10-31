# GraphPoolingGarden

- [GraphPoolingGarden](#graphpoolinggarden)
  - [项目介绍](#项目介绍)
  - [支持的数据集](#支持的数据集)
  - [支持的pooling方法](#支持的pooling方法)
  - [支持的readout方法](#支持的readout方法)
  - [用法](#用法)
    - [1. 创建配置文件](#1-创建配置文件)
    - [2. 运行train.py](#2-运行trainpy)


## 项目介绍

这是一个收集图池化方法，便于对比实验和后续改进的开源库。

## 支持的数据集
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

## 支持的pooling方法
- set2set
- sagpool(sequence/hierarchical)
- Graph U-net
- Diffpool

## 支持的readout方法
- mean
- max
- sum

## 用法

### 1. 创建配置文件
在configs文件夹下创建config.json文件，内容格式示例如下：

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

当键对应的值为列表时，将会通过排列组合将所有的组合都运行一次，并记录最终结果至以dataset_name_pooling命名的csv文件中。
目前支持列表的键：
- dataset_name
- seed

### 2. 运行train.py

`python graphpoolinggarden/train.py --config configs/config.json`
