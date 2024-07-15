# -*- coding: UTF-8 -*-

import torch.nn.functional as F
from torch import nn


class PreNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(PreNet, self).__init__()
        in_sizes = [in_features] + list(out_features[:-1])
        self.layers = nn.ModuleList([
            nn.Linear(in_size, out_size, bias=False) for (in_size, out_size) in zip(in_sizes, out_features)
        ])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x
