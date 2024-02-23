import math

import torch
import os
import time
import random
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, csgraph
from sklearn.metrics import roc_auc_score, classification_report
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from name import *
import batchdata

def set_seed(seed):
    if seed == 0:
        seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed

def load_data(data):
    path = os.path.join(DATADIR, data)

    graphlabel_path = os.path.join(path, data + NEWLABEL)
    graphlabels = np.loadtxt(graphlabel_path, dtype=np.int64)

    edge_path = os.path.join(path, data + ADJ)
    edges = np.loadtxt(edge_path, dtype=np.int64, delimiter=",")
    edges -= 1

    graphindicator_path = os.path.join(path, data + GRAPHIND)
    graphindicator = np.loadtxt(graphindicator_path, dtype=np.int64)

    _, graph_size = np.unique(graphindicator, return_counts=True)
    adj = csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(graphindicator.size, graphindicator.size))

    nodeattr_path = os.path.join(path, data + NODEATTR)
    nodeattrs = np.loadtxt(nodeattr_path, dtype=np.float, delimiter=",")

    adjs = []
    features = []
    idx = 0
    for i in range(graph_size.size):
        adjs.append(adj[idx:idx + graph_size[i], idx:idx + graph_size[i]])
        features.append(nodeattrs[idx:idx + graph_size[i], :])
        idx += graph_size[i]

    train_path = os.path.join(path, data + TRAIN)
    train_index = np.loadtxt(train_path, dtype=np.int64)

    val_path = os.path.join(path, data + VAL)
    val_index = np.loadtxt(val_path, dtype=np.int64)

    test_path = os.path.join(path, data + TEST)
    test_index = np.loadtxt(test_path, dtype=np.int64)
    return adjs, features, graphlabels, train_index, val_index, test_index

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def generate_batches(adjs, features, graphlabels, batchsize, shuffle):
    N = len(graphlabels)
    if shuffle:
        index = np.random.permutation(N)
    else:
        index = np.array(range(N), dtype=np.int32)

    batchs = []
    for i in range(0, N, batchsize):
        ngraph = min(i + batchsize, N) - i
        nnode = sum([adjs[index[j]].shape[0] for j in range(i, min(i + batchsize, N))])

        adj_batch = lil_matrix((nnode, nnode))
        features_batch = np.zeros((nnode, features[0].shape[1]))
        label_batch = np.zeros(ngraph)
        graphpool_batch = lil_matrix((ngraph, nnode))

        xLx_batch = torch.zeros((ngraph, features[0].shape[1]))

        idx = 0

        label_count = [0, 0]
        node_belong = []
        for j in range(i, min(i + batchsize, N)):
            n = adjs[index[j]].shape[0]
            adj_batch[idx:idx + n, idx:idx + n] = adjs[index[j]]
            features_batch[idx:idx + n, :] = features[index[j]]
            label_batch[j - i] = graphlabels[index[j]]
            graphpool_batch[j - i, idx:idx + n] = 1
            label_count[int(graphlabels[index[j]])] += 1
            node_belong.append(list(range(idx, idx + n)))
            
            temp_L = sparse_mx_to_torch_sparse_tensor(csgraph.laplacian(adjs[index[j]], normed=True))
            temp_x = torch.FloatTensor(features[index[j]])
            xLx_batch[j - i] = torch.diag(torch.mm(torch.mm(temp_x.T, temp_L.to_dense()), temp_x))

            idx += n

        adj_list = sparse_mx_to_torch_sparse_tensor(adj_batch)
        features_list = torch.FloatTensor(features_batch)
        label_list = torch.LongTensor(label_batch)
        graphpool_list = sparse_mx_to_torch_sparse_tensor(graphpool_batch)
        lap_list = sparse_mx_to_torch_sparse_tensor(csgraph.laplacian(adj_batch, normed=True))
        edge_index = from_scipy_sparse_matrix(adj_batch)[0]


        batchs.append(batchdata.Batch(adj_list, features_list, label_list, graphpool_list, lap_list, edge_index, label_count, node_belong, xLx_batch))
    return batchs

def compute_metrics(preds, truths):

    auc = roc_auc_score(truths.detach().cpu().numpy(), preds.detach().cpu().numpy()[:, 1])

    target_names = ['C0', 'C1']
    DICT = classification_report(truths.detach().cpu().numpy(), preds.detach().cpu().numpy().argmax(axis=1), target_names=target_names, output_dict=True)

    macro_f1 = DICT['macro avg']['f1-score']

    return auc, macro_f1

