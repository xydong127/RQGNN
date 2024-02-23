import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.loss import _Loss


def focal_loss(labels, logits, alpha, gamma):
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss

def CB_loss(labels, logits, samples_per_cls, nclass, beta, gamma):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    if effective_num[1] > 0:
        weights = (1.0 - beta) / np.array(effective_num)
    else:
        weights = np.array([(1.0 - beta) / effective_num[0], 0])
    weights = weights / np.sum(weights) * nclass

    labels_one_hot = F.one_hot(labels, nclass).float()

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, nclass)

    cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)

    return cb_loss
