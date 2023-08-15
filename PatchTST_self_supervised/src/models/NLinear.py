import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NLinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, seq_len, pred_len):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

        #self.Linear = nn.Linear(self.seq_len, self.seq_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last # seq_last is substracted from every element in the channel
        #x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x # [Batch, Output length, Channel]