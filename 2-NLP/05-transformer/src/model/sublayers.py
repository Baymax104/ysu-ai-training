# -*- coding: UTF-8 -*-
import math

import torch
import torch.nn.functional as F
from torch import nn


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout):
        """
        query(..., N, d_k)
        key(..., M, d_k)
        value(..., M, d_v)
        """
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        qk = torch.matmul(query, key.transpose(-1, -2))
        qk = qk / torch.sqrt(torch.tensor(d_k, dtype=torch.float))
        if mask is not None:
            qk = qk.to(torch.float32)
            qk = torch.masked_fill(qk, torch.eq(mask, 0), -1e9)
        qk = F.softmax(qk, dim=-1)
        qk = self.dropout(qk)
        context = torch.matmul(qk, value)
        return context


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, h, dropout):
        """
        input(batch_size, sequence_length, d_model)
        output(batch_size, sequence_length, d_model)
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_model = d_model
        self.num_heads = h
        self.d_k = self.d_v = d_model // h

        self.linear_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.linear_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.linear_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.attention = ScaledDotProductAttention(dropout)
        self.linear_out = nn.Linear(self.d_model, self.d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # (batch_size, sequence_length, d_model)
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)

        # 分割为多头
        # (batch_size, sequence_length, h, d_k)
        query = query.view(batch_size, -1, self.num_heads, self.d_k)
        key = key.view(batch_size, -1, self.num_heads, self.d_k)
        value = value.view(batch_size, -1, self.num_heads, self.d_v)

        # (batch_size, h, sequence_length, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        context = self.attention(query, key, value, mask)

        # (batch_size, sequence_length, d_model)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        context = self.linear_out(context)
        return context


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, max_length, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_length = max_length
        pe = torch.zeros(max_length, d_model)
        # (max_length, 1)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        # (d_model // 2,)
        dividend = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(position * dividend)
        pe[:, 1::2] = torch.cos(position * dividend)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x(batch_size, sequence_length, embedding_size)
        return self.pe[:, :x.size(1)]
