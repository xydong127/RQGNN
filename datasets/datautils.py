import time
import random
import numpy as np
import os
import torch
from sklearn.preprocessing import OneHotEncoder

from name import *

def set_seed(seed):
    if seed == 0:
        seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed

def gen_nodeattr(data):
    nodeattr_path = os.path.join(data, data + NODEATTR)
    if not os.path.exists(nodeattr_path):
        nodelabel_path = os.path.join(data, data + NODELABEL)
        nodelabels = np.loadtxt(nodelabel_path, dtype=np.int64).reshape(-1, 1)
        enc = OneHotEncoder(sparse=False)
        nodeattrs = enc.fit_transform(nodelabels)
        np.savetxt(nodeattr_path, nodeattrs, fmt='%f', delimiter=',')

def split_train_val_test(data, trainsz, testsz):
    graphlabel_path = os.path.join(data, data + GRAPHLABEL)
    graphlabels = np.loadtxt(graphlabel_path, dtype=np.int64)
    graphlabels = np.where(graphlabels == -1, 0, graphlabels)

    normalinds = []
    abnormalinds = []
    for i, label in enumerate(graphlabels):
        if label == 0:
            normalinds.append(i)
        elif label == 1:
            abnormalinds.append(i)
        else:
            print("Exist wrong label: {}, index: {}".format(label, i))
            print("Generate fail")
            exit()

    random.shuffle(normalinds)
    random.shuffle(abnormalinds)

    train_normal = np.array(normalinds[: int(trainsz * len(normalinds))])
    val_normal = np.array(normalinds[int(trainsz * len(normalinds)): int((1 - testsz) * len(normalinds))])
    test_normal = np.array(normalinds[int((1 - testsz) * len(normalinds)): ])

    train_abnormal = np.array(abnormalinds[: int(trainsz * len(abnormalinds))])
    val_abnormal = np.array(abnormalinds[int(trainsz * len(abnormalinds)): int((1 - testsz) * len(abnormalinds))])
    test_abnormal = np.array(abnormalinds[int((1 - testsz) * len(abnormalinds)):])

    train_index = np.concatenate((train_normal, train_abnormal))
    val_index = np.concatenate((val_normal, val_abnormal))
    test_index = np.concatenate((test_normal, test_abnormal))

    random.shuffle(train_index)
    random.shuffle(val_index)
    random.shuffle(test_index)

    print("Train size: {}, normal size: {}, abnormal size: {}".format(len(train_index), len(train_normal), len(train_abnormal)))
    print("Val size: {}, normal size: {}, abnormal size: {}".format(len(val_index), len(val_normal), len(val_abnormal)))
    print("Test size: {}, normal size: {}, abnormal size: {}".format(len(test_index), len(test_normal), len(test_abnormal)))

    print("Total size: {}, generate size: {}".format(len(graphlabels), len(train_index) + len(val_index) + len(test_index)))

    train_path = os.path.join(data, data + TRAIN)
    val_path = os.path.join(data, data + VAL)
    test_path = os.path.join(data, data + TEST)

    np.savetxt(train_path, train_index, fmt='%d')
    np.savetxt(val_path, val_index, fmt='%d')
    np.savetxt(test_path, test_index, fmt='%d')

    np.savetxt(os.path.join(data, data + NEWLABEL), graphlabels, fmt='%d')
