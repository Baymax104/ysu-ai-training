# -*- coding: UTF-8 -*-
import torch
from torch import nn

from sublayers import MultiHeadAttention, FeedForward


class EncoderLayer(nn.Module):

    def __init__(self, d_model, h, d_ff, dropout):
        """forward(x, mask)"""
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, h, dropout)
        self.feedforward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.attention(x, x, x, mask)
        x = self.norm1(x)
        x = x + self.feedforward(x)
        x = self.norm2(x)
        return x


class Encoder(nn.Module):

    def __init__(self, n, d_model, h, d_ff, dropout):
        """forward(input_emb, attention_mask)"""
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff, dropout) for _ in range(n)])

    def forward(self, input_emb, attention_mask):
        for layer in self.layers:
            input_emb = layer(input_emb, attention_mask)
        return input_emb


if __name__ == '__main__':
    x = torch.arange(20, dtype=torch.float).view(1, 5, 4)
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float)
    encoder_layer = EncoderLayer(4, 2, 16, 0.1)
    output = encoder_layer(x, mask)
    print(f'Encoder Layer: {output.size()}')

if __name__ == '__main__':
    x = torch.arange(20, dtype=torch.float).view(1, 5, 4)
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float)
    encoder = Encoder(2, 4, 2, 16, 0.1)
    result = encoder(x, mask)
    print(f'Encoder: {result.size()}')
