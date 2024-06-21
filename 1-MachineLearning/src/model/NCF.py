# -*- coding: UTF-8 -*-
import torch
from torch import nn


class NCF(nn.Module):

    def __init__(self, user_num, item_num):
        super().__init__()
        self.gmf_user_emb = nn.Embedding(user_num, 128)
        self.gmf_item_emb = nn.Embedding(item_num, 128)
        self.gmf_linear = nn.Linear(128, 64, bias=False)

        self.mlp_user_emb = nn.Embedding(user_num, 128)
        self.mlp_item_emb = nn.Embedding(item_num, 128)
        self.mlp = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
        )

        self.ncf_linear = nn.Linear(64 + 64, 1)

    def forward(self, user_id, item_id):
        gmf_user_emb = self.gmf_user_emb(user_id)
        gmf_item_emb = self.gmf_item_emb(item_id)
        gmf_output = self.gmf_linear(gmf_user_emb * gmf_item_emb)

        mlp_user_emb = self.mlp_user_emb(user_id)
        mlp_item_emb = self.mlp_item_emb(item_id)
        mlp_input = torch.cat([mlp_user_emb, mlp_item_emb], dim=1)
        mlp_output = self.mlp(mlp_input)

        ncf_input = torch.cat([gmf_output, mlp_output], dim=1)
        return self.ncf_linear(ncf_input)
