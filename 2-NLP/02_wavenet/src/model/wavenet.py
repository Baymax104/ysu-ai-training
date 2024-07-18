# -*- coding: UTF-8 -*-
from torch import nn

from block import ResidualBlock
from layers import OneConv, OutputLayer


class WaveNet(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 k_layer=10,
                 blocks=4,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=256,
                 kernel_size=2):
        super(WaveNet, self).__init__()

        self.start_conv = OneConv(in_channels=in_channels, out_channels=dilation_channels)

        self.blocks = [
            ResidualBlock(in_channels=dilation_channels, residual_channels=residual_channels,
                          skip_channels=skip_channels, kernel_size=kernel_size, k_layer=k_layer)
            for _ in range(blocks)
        ]
        self.blocks = nn.ModuleList(self.blocks)

        self.output_layer = OutputLayer(
            in_channels=skip_channels,
            hidden_channels=skip_channels,
            out_channels=out_channels
        )

    def forward(self, x):
        x = self.start_conv(x)
        total_skip = 0
        for block in self.blocks:
            residual, skip = block(x)
            total_skip = total_skip + skip
            x = residual
        output = self.output_layer(total_skip)
        return output
