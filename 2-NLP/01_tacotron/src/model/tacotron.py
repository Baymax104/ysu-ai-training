# -*- coding: UTF-8 -*-
from torch import nn

from .cbhg import CBHG
from .decoder import Decoder
from .encoder import Encoder


class Tacotron(nn.Module):

    def __init__(self, num_chars, batch_size):
        super(Tacotron, self).__init__()

        self.char_embedding = nn.Embedding(num_chars, 256, padding_idx=0)
        self.char_embedding.weight.data.normal_(0, 0.3)

        self.encoder = Encoder(in_features=256)
        self.decoder = Decoder(in_features=256, frame_features=80, r=3, batch_size=batch_size)

        self.postnet = CBHG(
            in_features=80,
            K=8,
            conv_bank_features=128,
            conv_projections=(256, 80),
            highway_features=128,
            gru_features=128,
            num_highways=4
        )

        self.linear = nn.Linear(in_features=256, out_features=1024)

    def forward(self, text):
        # text(batch, time_step)
        # (batch, time_step, 256)
        x = self.char_embedding(text)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.postnet(x)
        x = self.linear(x)
        return x
