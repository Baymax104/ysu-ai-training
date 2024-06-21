# -*- coding: UTF-8 -*-
import torch
from torch import nn


class MF(nn.Module):

    def __init__(self, user_num, item_num):
        super().__init__()
        self.user_embedding = nn.Embedding(user_num, 128)
        self.item_embedding = nn.Embedding(item_num, 128)

        self.user_bias = nn.Embedding(user_num, 1)
        self.item_bias = nn.Embedding(item_num, 1)

        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_id, item_id):
        user_embedding = self.user_embedding(user_id)
        item_embedding = self.item_embedding(item_id)
        user_bias = self.user_bias(user_id).squeeze()
        item_bias = self.item_bias(item_id).squeeze()
        return (user_embedding * item_embedding).sum(dim=1) + user_bias + item_bias + self.global_bias
