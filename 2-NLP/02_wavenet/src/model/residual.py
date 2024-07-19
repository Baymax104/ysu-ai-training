# -*- coding: UTF-8 -*-
import torch.nn.functional as F
from torch import nn

from .layers import OneConv


class Residual(nn.Module):

    def __init__(self, in_channels, residual_channels, skip_channels):
        super(Residual, self).__init__()
        self.filter_conv = OneConv(in_channels=in_channels, out_channels=residual_channels)
        self.gate_conv = OneConv(in_channels=in_channels, out_channels=residual_channels)
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
