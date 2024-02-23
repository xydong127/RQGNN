import argparse
import time
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

import utils
import model
from name import *
import lossfunc

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='MCF-7', help='Dataset used')
parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
parser.add_argument('--batchsize', type=int, default=512, help='Training batch size')
parser.add_argument('--nepoch', type=int, default=100, help='Number of training epochs')
parser.add_argument('--hdim', type=int, default=64, help='Hidden feature dim')
parser.add_argument('--width', type=int, default=4, help='Width of GCN')
parser.add_argument('--depth', type=int, default=6, help='Depth of GCN')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
parser.add_argument('--normalize', type=int, default=1, help='Whether batch normalize')
parser.add_argument('--beta', type=float, default=0.999, help='CB loss beta')
parser.add_argument('--gamma', type=float, default=1.5, help='CB loss gamma')
parser.add_argument('--decay', type=float, default=0, help='Weight decay')
parser.add_argument('--seed', type=int, default=10, help='Random seed')
parser.add_argument('--patience', type=int, default=50, help='Patience')
args = parser.parse_args()

data = args.data
lr = args.lr
batchsize = args.batchsize
nepoch = args.nepoch
hdim = args.hdim
width = args.width
depth = args.depth
dropout = args.dropout
normalize = args.normalize
beta = args.beta
gamma = args.gamma
decay = args.decay
seed = args.seed
patience = args.patience

nclass = 2

utils.set_seed(seed)

print("Model info:")
print(json.dumps(args.__dict__, indent='\t'))

adjs, features, graphlabels, train_index, val_index, test_index = utils.load_data(data)
featuredim = features[0].shape[1]

adj_train = [adjs[i] for i in train_index]
feats_train = [features[i] for i in train_index]
label_train = [graphlabels[i] for i in train_index]

adj_val = [adjs[i] for i in val_index]
feats_val = [features[i] for i in val_index]
label_val = [graphlabels[i] for i in val_index]

adj_test = [adjs[i] for i in test_index]
feats_test = [features[i] for i in test_index]
label_test = [graphlabels[i] for i in test_index]

ny_0 = label_train.count(0)
ny_1 = label_train.count(1)

gad = model.GADGNN(featuredim, hdim, nclass, width, depth, dropout, normalize)
optimizer = optim.Adam(gad.parameters(), lr=lr, weight_decay=decay)

bestauc = 0
bestf1 = 0
bestepochauc = 0
bestepochf1 = 0
bestmodelauc = deepcopy(gad)
bestmodelf1 = deepcopy(gad)

patiencecount = 0

print("Starts training...")
for epoch in range(nepoch):
    epoch_start = time.time()
    gad.train()
    train_batches = utils.generate_batches(adj_train, feats_train, label_train, batchsize, True)
    epoch_loss = 0


    for train_batch in train_batches:
        optimizer.zero_grad()
        outputs = gad(train_batch)
        loss = lossfunc.CB_loss(train_batch.label_list, outputs, [ny_0, ny_1], nclass, beta, gamma)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_end = time.time()
    print('Epoch: {}, loss: {}, time cost: {}'.format(epoch, epoch_loss / len(train_batches), epoch_end - epoch_start))

    gad.eval()
    val_batches = utils.generate_batches(adj_val, feats_val, label_val, batchsize, False)
    preds = torch.Tensor()
    truths = torch.Tensor()
    for i, val_batch in enumerate(val_batches):
        outputs = gad(val_batch)
        outputs = nn.functional.softmax(outputs, dim=1)
        if i == 0:
            preds = outputs
            truths = val_batch.label_list
        else:
            preds = torch.cat((preds, outputs), dim=0)
            truths = torch.cat((truths, val_batch.label_list), dim=0)

    auc_val, f1_score_val = utils.compute_metrics(preds, truths)
    print("Val auc: {}, f1: {}".format(auc_val, f1_score_val))

    if bestauc <= auc_val:
        bestauc = auc_val
        bestepochauc = epoch
        bestmodelauc = deepcopy(gad)

    if bestf1 <= f1_score_val:
        patiencecount = 0
        bestf1 = f1_score_val
        bestepochf1 = epoch
        bestmodelf1 = deepcopy(gad)
    else:
        patiencecount += 1

    if patiencecount > patience:
        break

print("Under the condition of auc, best idx: {}".format(bestepochauc))
test_batches = utils.generate_batches(adj_test, feats_test, label_test, batchsize, False)
preds = torch.Tensor()
truths = torch.Tensor()
for i, test_batch in enumerate(test_batches):
    outputs = bestmodelauc(test_batch)
    outputs = nn.functional.softmax(outputs, dim=1)
    if i == 0:
        preds = outputs
        truths = test_batch.label_list
    else:
        preds = torch.cat((preds, outputs), dim=0)
        truths = torch.cat((truths, test_batch.label_list), dim=0)

auc_test, f1_score_test = utils.compute_metrics(preds, truths)
print("Test auc: {}, f1: {}".format(auc_test, f1_score_test))

print("Under the condition of f1, best idx: {}".format(bestepochf1))
test_batches = utils.generate_batches(adj_test, feats_test, label_test, batchsize, False)
preds = torch.Tensor()
truths = torch.Tensor()
for i, test_batch in enumerate(test_batches):
    outputs = bestmodelf1(test_batch)
    outputs = nn.functional.softmax(outputs, dim=1)
    if i == 0:
        preds = outputs
        truths = test_batch.label_list
    else:
        preds = torch.cat((preds, outputs), dim=0)
        truths = torch.cat((truths, test_batch.label_list), dim=0)

auc_test, f1_score_test = utils.compute_metrics(preds, truths)
print("Test auc: {}, f1: {}".format(auc_test, f1_score_test))
