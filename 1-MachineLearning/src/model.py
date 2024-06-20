# -*- coding: UTF-8 -*-
import torch
from torch import nn


class AnimeMF(nn.Module):

    def __init__(self, user_num, anime_num):
        super().__init__()
        self.user_embedding = nn.Embedding(user_num, 128)
        self.anime_embedding = nn.Embedding(anime_num, 128)

        self.user_bias = nn.Embedding(user_num, 1)
        self.anime_bias = nn.Embedding(anime_num, 1)

        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_id, anime_id):
        user_embedding = self.user_embedding(user_id)
        anime_embedding = self.anime_embedding(anime_id)
        user_bias = self.user_bias(user_id).squeeze()
        anime_bias = self.anime_bias(anime_id).squeeze()
        return (user_embedding * anime_embedding).sum(dim=1) + user_bias + anime_bias + self.global_bias
