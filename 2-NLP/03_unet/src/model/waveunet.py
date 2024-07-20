# -*- coding: UTF-8 -*-
import torch
import torch.nn.functional as F
from torch import nn

from .sampling import Downsampling, Upsampling, center_crop


class WaveUNet(nn.Module):

    def __init__(self, in_channels, num_source, layers=12, Fc=24):
        super(WaveUNet, self).__init__()
        self.in_channels = in_channels
        self.num_source = num_source
        self.layers = layers

        self.downsampling_layers = nn.ModuleList()
        self.upsampling_layers = nn.ModuleList()
        self.bottleneck = nn.Conv1d(
            in_channels=Fc * layers,
            out_channels=Fc * (layers + 1),
            kernel_size=15,
            stride=1,
            padding=7
        )
        self.end_conv = nn.Conv1d(
            in_channels=in_channels * 2,
            out_channels=in_channels * num_source,
            kernel_size=1,
            stride=1,
            padding=0
        )

        for l in range(1, layers + 1):
            downsampling = Downsampling(
                in_channels=in_channels if l == 1 else Fc * (l - 1),
                shortcut_channels=Fc * l,
                out_channels=Fc * l,
                activation=F.leaky_relu if l != layers else F.tanh,
                kernel_size=15
            )

            upsampling = Upsampling(
                in_channels=Fc * l if l != layers else Fc * (l + 1),
                shortcut_channels=Fc * l,
                out_channels=in_channels if l == 1 else Fc * (l - 1),
                kernel_size=5
            )

            self.downsampling_layers.append(downsampling)
            self.upsampling_layers.append(upsampling)

    def forward(self, x):
        inputs = x
        shortcuts = []
        for l in range(self.layers):
            x, shortcut = self.downsampling_layers[l](x)
            shortcuts.append(shortcut)
        x = self.bottleneck(x)
        for l in range(self.layers - 1, -1, -1):
            x = self.upsampling_layers[l](x, shortcuts[l])

        x = center_crop(x, inputs)
        x = torch.cat([x, inputs], dim=1)
        x = self.end_conv(x)
        source_outputs = x.split(self.in_channels, dim=1)
        # assert len(source_outputs) == self.num_source
        return source_outputs
