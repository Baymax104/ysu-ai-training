# -*- coding: UTF-8 -*-
import torch
from torch import nn


class MLP(nn.Module):

    def __init__(self, user_num, item_num):
        super().__init__()
        self.user_embedding = nn.Embedding(user_num, 128)
        self.item_embedding = nn.Embedding(item_num, 128)
        self.linear_layer = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, user_id, item_id):
        user_emb = self.user_embedding(user_id)
        item_emb = self.item_embedding(item_id)
        x = torch.cat((user_emb, item_emb), dim=1)
        x = self.linear_layer(x)
        return x.squeeze()
