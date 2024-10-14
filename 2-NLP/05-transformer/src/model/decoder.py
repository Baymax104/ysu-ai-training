# -*- coding: UTF-8 -*-
import torch
from torch import nn

from sublayers import MultiHeadAttention, FeedForward


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


if __name__ == '__main__':
    x = torch.arange(20, dtype=torch.float).view(1, 5, 4)
    encoder_output = torch.randn([1, 5, 4], dtype=torch.float)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float)
    causal_mask = torch.tril(torch.ones(5, 5, dtype=torch.float)).unsqueeze(0)
    decoder_layer = DecoderLayer(4, 2, 16, 0.1)
    result = decoder_layer(x, encoder_output, attention_mask, causal_mask)
    print(f'Decoder Layer: {result.size()}')

if __name__ == '__main__':
    x = torch.arange(20, dtype=torch.float).view(1, 5, 4)
    encoder_output = torch.randn([1, 5, 4], dtype=torch.float)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float)
    causal_mask = torch.tril(torch.ones(5, 5, dtype=torch.float)).unsqueeze(0)
    decoder = Decoder(2, 4, 2, 16, 0.1)
    result = decoder(x, encoder_output, attention_mask, causal_mask)
    print(f'Decoder: {result.size()}')
