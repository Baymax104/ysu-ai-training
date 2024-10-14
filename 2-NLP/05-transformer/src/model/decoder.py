# -*- coding: UTF-8 -*-
from torch import nn

from .sublayers import MultiHeadAttention, FeedForward


class DecoderLayer(nn.Module):

    def __init__(self, d_model, h, d_ff, dropout):
        """forward(x, encoder_output, attention_mask, causal_mask)"""
        super(DecoderLayer, self).__init__()
        self.masked_attention = MultiHeadAttention(d_model, h, dropout)
        self.attention = MultiHeadAttention(d_model, h, dropout)
        self.feedforward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, attention_mask, causal_mask):
        x = x + self.masked_attention(x, x, x, causal_mask)
        x = self.norm1(x)
        x = x + self.attention(x, encoder_output, encoder_output, attention_mask)
        x = self.norm2(x)
        x = x + self.feedforward(x)
        x = self.norm3(x)
        return x


class Decoder(nn.Module):

    def __init__(self, n, d_model, h, d_ff, dropout):
        """forward(output_emb, encoder_output, attention_mask, causal_mask)"""
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, h, d_ff, dropout) for _ in range(n)])

    def forward(self, output_emb, encoder_output, attention_mask, causal_mask):
        for layer in self.layers:
            output_emb = layer(output_emb, encoder_output, attention_mask, causal_mask)
        return output_emb
