# -*- coding: UTF-8 -*-
from torch import nn


class GMF(nn.Module):

    def __init__(self, user_num, item_num):
        super().__init__()
        self.user_embedding = nn.Embedding(user_num, 128)
        self.item_embedding = nn.Embedding(item_num, 128)
        self.linear = nn.Linear(128, 1)

    def forward(self, user_id, item_id):
        user_emb = self.user_embedding(user_id)
        item_emb = self.item_embedding(item_id)
        mf = user_emb * item_emb
        pred = self.linear(mf)
        return pred.squeeze()
