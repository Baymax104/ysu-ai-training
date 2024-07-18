# -*- coding: UTF-8 -*-
from torch import nn


class CausalConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) * dilation + 1,
            dilation=dilation
        )

    def forward(self, x):
        x = self.conv(x)
        x = x[:, :, :-self.causal_conv.padding[0]]
        return x
