# -*- coding: UTF-8 -*-
from torch import nn
import torch.nn.functional as F
from layers import OneConv


class Residual(nn.Module):

    def __init__(self, in_channels, residual_channels, skip_channels, kernel_size):
        super(Residual, self).__init__()
        self.filter_conv = nn.Conv1d(in_channels=in_channels, out_channels=residual_channels, kernel_size=kernel_size)
        self.gate_conv = nn.Conv1d(in_channels=in_channels, out_channels=residual_channels, kernel_size=kernel_size)
        self.residual_conv = OneConv(in_channels=residual_channels, out_channels=in_channels)
        self.skip_conv = OneConv(in_channels=residual_channels, out_channels=skip_channels)


    def forward(self, x):
        fil = F.tanh(self.filter_conv(x))
        gate = F.sigmoid(self.gate_conv(x))
        z = fil * gate
        skip = self.skip_conv(z)
        residual = self.residual_conv(z)
        residual += x
        return residual, skip
