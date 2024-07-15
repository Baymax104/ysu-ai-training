# -*- coding: UTF-8 -*-
import torch
import torch.nn.functional as F
from torch import nn


class BatchNormConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=None):
        super(BatchNormConv1d, self).__init__()
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1d(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class HighwayNet(nn.Module):
    def __init__(self, in_size, out_size):
        super(HighwayNet, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()  # bias设为0
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)  # bias设为-1

    def forward(self, inputs):
        H = F.relu(self.H(inputs))
        T = F.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


class CBHG(nn.Module):
    r"""
    Shapes:
        - inputs: (batch, in_features, time_step)
        - outputs: (batch, time_step, gru_features * 2)
    """

    def __init__(self,
                 in_features,
                 K=16,
                 conv_bank_features=128,
                 conv_projections=(128, 128),
                 highway_features=128,
                 gru_features=128,
                 num_highways=4):
        super(CBHG, self).__init__()
        self.in_features = in_features
        self.K = K
        self.conv_bank_features = conv_bank_features
        self.conv_projections = conv_projections
        self.highway_features = highway_features
        self.gru_features = gru_features
        self.num_highways = num_highways

        self.conv1d_banks = nn.ModuleList([
            BatchNormConv1d(
                in_channels=in_features,
                out_channels=conv_bank_features,
                kernel_size=k,
                stride=1,
                padding=((k - 1) // 2, k // 2),
                activation=nn.ReLU()
            ) for k in range(1, K + 1)
        ])

        bank_out_features = [K * conv_bank_features] + list(conv_projections[:-1])
        self.conv1d_projections = nn.ModuleList([
            BatchNormConv1d(
                in_channels=in_size,
                out_channels=out_size,
                kernel_size=3,
                stride=1,
                padding=(1, 1),
                activation=nn.ReLU()
            ) for (in_size, out_size) in zip(bank_out_features, conv_projections)
        ])

        if highway_features != conv_projections[-1]:
            self.linear = nn.Linear(conv_projections[-1], highway_features, bias=False)

        self.highways = nn.ModuleList([
            HighwayNet(
                in_size=highway_features,
                out_size=highway_features
            ) for _ in range(num_highways)
        ])

        self.gru = nn.GRU(
            input_size=gru_features,
            hidden_size=gru_features,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, inputs):
        # (batch, features, time_step)
        x = inputs
        bank_outputs = []
        # conv1d bank
        # (batch, features * K, time_step)
        for conv1d in self.conv1d_banks:
            out = conv1d(x)
            bank_outputs.append(out)
        x = torch.cat(bank_outputs, dim=1)  # stacking

        assert x.size(1) == self.conv_bank_features * self.K

        # conv1d projections
        # (batch, features, time_step)
        for conv1d in self.conv1d_projections:
            x = conv1d(x)

        x += inputs  # residual
        # (batch, time_step, features)
        x = x.transpose(1, 2)

        if self.highway_features != self.conv_projections[-1]:
            x = self.linear(x)

        # highways
        # (batch, time_step, features)
        for highway in self.highways:
            x = highway(x)

        # gru
        # (batch, time_step, features * 2)
        self.gru.flatten_parameters()
        outputs, _ = self.gru(x)
        return outputs
