# -*- coding: UTF-8 -*-
import torch
import torch.nn.functional as F
from torch import nn


def center_crop(x, shortcut):
    """central crop x tensor in time_step dimension to fix shortcut time steps"""
    assert x.size(dim=-1) >= shortcut.size(dim=-1)
    diff = x.size(dim=-1) - shortcut.size(dim=-1)
    if diff == 0:
        return x
    assert diff % 2 == 0
    r = diff // 2
    return x[:, :, r:-r]


class Downsampling(nn.Module):

    def __init__(self, in_channels, shortcut_channels, out_channels, activation=F.leaky_relu, kernel_size=15):
        super(Downsampling, self).__init__()
        self.activation = activation

        self.pre_shortcut_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=shortcut_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,  # keep unchanged time steps
        )

        self.post_shortcut_conv = nn.Conv1d(
            in_channels=shortcut_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,  # keep unchanged time steps
        )

        # time_step down sampling, time_step = time_step / 2
        self.downsampling = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
        )

    def forward(self, x):
        x = self.pre_shortcut_conv(x)
        x = self.activation(x)
        shortcut = x
        x = self.post_shortcut_conv(x)
        x = self.downsampling(x)
        return x, shortcut


class Upsampling(nn.Module):

    def __init__(self, in_channels, out_channels, shortcut_channels, kernel_size=5):
        super(Upsampling, self).__init__()

        # self.upsampling uses linear F.interpolate

        self.pre_shortcut_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=shortcut_channels,
            stride=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )

        self.post_shortcut_conv = nn.Conv1d(
            in_channels=shortcut_channels * 2,
            out_channels=out_channels,
            stride=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, x, shortcut):
        # upsampling
        x = F.interpolate(x, scale_factor=2, mode='linear')
        x = self.pre_shortcut_conv(x)
        # crop
        x = center_crop(x, shortcut)
        # concatenate
        x = torch.cat([x, shortcut], dim=1)
        x = self.post_shortcut_conv(x)
        return x
