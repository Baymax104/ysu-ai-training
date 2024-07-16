# -*- coding: UTF-8 -*-
from torch import nn


class StopNet(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features, 1)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain("linear"))

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        return x
