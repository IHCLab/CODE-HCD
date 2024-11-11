import torch
import torch.nn as nn
import torch.nn.functional as F
from models import layers
# from layers import GraphAttentionLayer
# from torch_geometric.nn import knn_graph, EdgeConv

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, A, nout, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.A = A
        self.attentions = [layers.GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = layers.GraphAttentionLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, self.A) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, self.A))
        return x
        
        
   
class Net(nn.Module):
    def __init__(self, channel: int, class_num: int, Q: torch.Tensor, A: torch.Tensor, model='normal'):
        super(Net, self).__init__()
        self.Q = Q
        self.A = A
        self.model=model
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  
        self.channel = channel
        
        self.GATNet1=GAT(nfeat=channel, nhid=64, A=A, nout=64, dropout=0.4, nheads=2, alpha=0.2)
        
        self.group = nn.Sequential()
        self.group.add_module('feature_extractor', torch.nn.Conv2d(128+2*channel, 128, kernel_size=(5,5), padding=2, groups=2))
        
        self.conv = nn.Sequential()
        self.conv.add_module('Conv', torch.nn.Conv2d(128, 64, kernel_size=(3,3), padding=1))

        self.linear1 = nn.Linear(128, 64)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)

        self.Softmax_linear =nn.Sequential(nn.Linear(64, class_num))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        (h, w, c) = x1.shape
        
        x1_flatten=x1.reshape([h * w, -1])
        x2_flatten=x2.reshape([h * w, -1])
        superpixels_flatten1 = torch.mm(self.norm_col_Q.t(), x1_flatten)
        superpixels_flatten2 = torch.mm(self.norm_col_Q.t(), x2_flatten)

        H1 = superpixels_flatten1
        H1 = self.GATNet1(H1)
        
        
        H2 = superpixels_flatten2
        H2 = self.GATNet1(H2)
        

        feature_map1 = torch.matmul(self.Q, H1)
        feature_map2 = torch.matmul(self.Q, H2)
        
        x = torch.cat((feature_map1, x1_flatten, feature_map2, x2_flatten), -1)
        
        x1 = self.group(torch.unsqueeze(x.reshape([h, w, 128+2*self.channel]).permute([2, 0, 1]), 0))
        x1 = torch.squeeze(x1, 0).permute([1, 2, 0]).reshape([h * w, -1])
        x2 = self.act1(self.bn1(x1))
        x3 = self.conv(torch.unsqueeze(x2.reshape([h, w, 128]).permute([2, 0, 1]), 0))
        x3 = torch.squeeze(x3, 0).permute([1, 2, 0]).reshape([h * w, -1])
        x4 = self.act1(self.bn2(x3))
        Y = self.Softmax_linear(x4)
        Y = F.softmax(Y, -1)
        
        return Y
    
    
