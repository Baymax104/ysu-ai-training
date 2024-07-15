# -*- coding: UTF-8 -*-
import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):

    def __init__(self, query_dim, embedding_dim, attention_dim):
        super(Attention, self).__init__()
        self.query_dim = query_dim
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim

        self.inputs_layer = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_layer = nn.Linear(query_dim, attention_dim, bias=False)
        self.attention = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, inputs, query):
        inputs = self.inputs_layer(inputs)  # (batch_size, time_steps, attention_dim)
        query = self.query_layer(query).unsqueeze(dim=1)  # (batch_size, 1, attention_dim)
        attention_scores = self.attention(torch.tanh(inputs + query)).squeeze(dim=-1)  # (batch_size, time_steps)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, time_steps)
        context = torch.bmm(attention_weights.unsqueeze(1), inputs).squeeze(dim=1)  # (batch_size, attention_dim)
        return context
