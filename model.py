import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch
from torch_geometric.nn import ChebConv


class GADGNN(nn.Module):
    def __init__(self, featuredim, hdim, nclass, width, depth, dropout, normalize):
        super(GADGNN, self).__init__()

        self.conv = []
        for i in range(width):
            self.conv.append(ChebConv(featuredim, featuredim, depth))

        self.linear = nn.Linear(featuredim, featuredim)
        self.linear2 = nn.Linear(featuredim, featuredim)
        self.linear3 = nn.Linear(featuredim*len(self.conv), hdim)
        self.linear4 = nn.Linear(hdim, hdim)
        self.act = nn.LeakyReLU()
        #self.act = nn.ReLU()


        self.linear5 = nn.Linear(featuredim, hdim)
        self.linear6 = nn.Linear(hdim, hdim)
        
        self.linear7 = nn.Linear(hdim * 2, nclass)
        #self.linear7 = nn.Linear(hdim, nclass)

        #self.attpool = nn.Linear(hdim, 1)

        self.bn = torch.nn.BatchNorm1d(hdim * 2)
        #self.bn = torch.nn.BatchNorm1d(hdim)

        self.dp = nn.Dropout(p=dropout)
        self.normalize = normalize

        self.linear8 = nn.Linear(featuredim, hdim)
        self.linear9 = nn.Linear(hdim, hdim)

    def forward(self, data):
        h = self.linear(data.features_list)
        h = self.act(h)

        h = self.linear2(h)
        h = self.act(h)

        h_final = torch.zeros([len(data.features_list), 0])
        for conv in self.conv:
            h0 = conv(h, data.edge_index)
            h_final = torch.cat([h_final, h0], -1)

        h = self.linear3(h_final)
        h = self.act(h)
        
        h = self.linear4(h)
        h = self.act(h)


        tmpscores = self.linear8(data.xLx_batch)
        tmpscores = self.act(tmpscores)
        tmpscores = self.linear9(tmpscores)
        tmpscores = self.act(tmpscores)
        scores = torch.zeros([len(data.features_list), 1])
        for i, node_belong in enumerate(data.node_belong):
            scores[node_belong] = torch.unsqueeze(torch.mv(h[node_belong], tmpscores[i]), 1)


        temp = torch.mul(data.graphpool_list.to_dense().T, scores).T

        h = torch.mm(temp, h)
        #h = torch.spmm(data.graphpool_list, h)



        xLx = self.linear5(data.xLx_batch)
        
        xLx = self.linear6(xLx)
        xLx = self.act(xLx)

        h = torch.cat([h, xLx], -1)

        if self.normalize:
            h = self.bn(h)

        h = self.dp(h)
        embeddings = self.linear7(h)

        return embeddings
