# -*- coding: UTF-8 -*-
from torch import nn
from causal_conv import CausalConv
from residual import Residual


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, residual_channels, skip_channels, kernel_size, k_layer):
        super(ResidualBlock, self).__init__()

        self.layers = nn.ModuleList()
        for k in range(k_layer):
            dilation = 2 ** k
            causal_conv = CausalConv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                dilation=dilation
            )
            residual_block = Residual(
                in_channels=in_channels,
                residual_channels=residual_channels,
                skip_channels=skip_channels,
                kernel_size=kernel_size
            )
            layer = nn.Sequential(causal_conv, residual_block)
            self.layers.append(layer)

    def forward(self, x):
        total_skip = 0
        for layer in self.layers:
            residual, skip = layer(x)
            total_skip = total_skip + skip
            x = residual
        return x, total_skip
