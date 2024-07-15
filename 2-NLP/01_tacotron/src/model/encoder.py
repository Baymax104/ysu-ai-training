# -*- coding: UTF-8 -*-
from torch import nn

from .cbhg import CBHG
from .prenet import PreNet


class Encoder(nn.Module):
    r"""
    Shapes:
        - inputs: (batch, time_step, in_features)
        - outputs: (batch, time_step, 128 * 2)
    """

    def __init__(self, in_features):
        super(Encoder, self).__init__()

        self.pre_net = PreNet(
            in_features=in_features,
            out_features=(256, 128)
        )

        self.cbhg = CBHG(
            in_features=128,
            K=16,
            conv_bank_features=128,
            conv_projections=(128, 128),
            highway_features=128,
            gru_features=128,
            num_highways=4
        )

    def forward(self, x):
        # (batch, time_step, 128)
        x = self.pre_net(x)
        # (batch, time_step, 256)
        x = self.cbhg(x.transpose(1, 2))
        return x
