
# Rayleigh Quotient Graph Neural Networks for Graph-level Anomaly Detection

## Tested Environment

- Ubuntu 20.04
- [Python 3.7.15](https://www.anaconda.com/products/individual#Downloads)
- [Sklearn 1.0.2](https://scikit-learn.org/stable/install.html)
- [Pytorch 1.12.1](https://pytorch.org/get-started/locally/#linux-installation)
- [Numpy 1.21.6](https://numpy.org/install/)
- [Torch_geometric 2.2.0](https://pytorch-geometric.readthedocs.io/en/latest/index.html)
- [Scipy 1.7.3](https://scipy.org/)

## Datasets

Download zip files from [TUDatasets](https://chrsmrrs.github.io/datasets/) and unzip them in datasets/. 

**Directory Structure**

```
├── datasets
│   ├── MCF-7
│   │   ├── MCF-7_A.txt
│   │	├── MCF-7_graph_indicator.txt
│   │	├── MCF-7_graph_labels.txt
│   │	├── MCF-7_node_labels.txt
│	├── dataset.py  
│	├── datautils.py
│	├── name.py
```
Use dataset.py to split train, val, and test sets. 

**Example**
```
python dataset.py --data MCF-7 --trainsz 0.7 --testsz 0.15
```

## Experiments

**Parameters**
- data: dataset name, default = 'MCF-7'
- lr: learning rate, default = 5e-3
- batchsize: batch size, default = 512 
- nepoch: number of training epochs, default = 100
- hdim: hidden dimension of RQGNN, default = 64
- width: width of RQGNN, default = 4
- depth: depth of RQGNN, default = 6
- dropout: dropout rate, default = 0.4
- normalize: batch normalize, default = 1
- beta: hyperparameter in loss function, default = 0.999
- gamma: hyperparameter in loss function, default = 1.5
- decay: weight decay, default = 0
- seed: random seed, default = 10
- patience: patience for training, default = 50

**Example**
```
python main.py --data MCF-7 --lr 5e-3 --batchsize 512 --nepoch 100 --hdim 64 --width 4 --depth 6 --dropout 0.4 --normalize 1 --beta 0.999 --gamma 1.5 --decay 0 --seed 10 --patience 50
```

## Citation

```
@inproceedings{rqgnn,
author = {Xiangyu, Dong and Xingyi, Zhang and Sibo, Wang},
title = {Rayleigh Quotient Graph Neural Networks for Graph-level Anomaly Detection},
year = {2024},
booktitle = {ICLR},
}
```
