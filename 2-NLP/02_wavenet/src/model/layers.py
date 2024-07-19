# -*- coding: UTF-8 -*-
from torch import nn



class OneConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OneConv, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=1,
            padding=0,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):
        return self.conv(x)


class OutputLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(OutputLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            OneConv(in_channels=in_channels, out_channels=hidden_channels),
            nn.ReLU(inplace=True),
            OneConv(in_channels=hidden_channels, out_channels=out_channels),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # (batch, skip_channel, time_step)
        return self.layer(x)
