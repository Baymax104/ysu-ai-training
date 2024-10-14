# -*- coding: UTF-8 -*-
import torch.nn.functional as F
from torch import nn

from .decoder import Decoder
from .encoder import Encoder
from .sublayers import PositionalEncoding


class Transformer(nn.Module):

    def __init__(self, num_embeddings, max_length, n, d_model, h, d_ff, dropout):
        """forward(input_ids, output_ids, input_attention_mask, output_attention_mask, causal_mask)"""
        super(Transformer, self).__init__()
        self.input_embedding = nn.Embedding(num_embeddings, d_model)
        self.output_embedding = nn.Embedding(num_embeddings, d_model)
        self.positional_encoding = PositionalEncoding(max_length, d_model)
        self.encoder = Encoder(n, d_model, h, d_ff, dropout)
        self.decoder = Decoder(n, d_model, h, d_ff, dropout)
        self.linear = nn.Linear(d_model, num_embeddings)

    def forward(self, input_ids, output_ids, input_attention_mask, output_attention_mask, causal_mask):
        input_emb = self.input_embedding(input_ids)
        input_emb = input_emb + self.positional_encoding(input_emb)
        output_emb = self.output_embedding(output_ids)
        output_emb = output_emb + self.positional_encoding(output_emb)
        encoder_output = self.encoder(input_emb, input_attention_mask)
        decoder_output = self.decoder(output_emb, encoder_output, output_attention_mask, causal_mask)
        result = F.softmax(self.linear(decoder_output), dim=-1)
        return result
