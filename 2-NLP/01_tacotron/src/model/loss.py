# -*- coding: UTF-8 -*-
import torch
from torch import nn


class MaskL1Loss(nn.Module):
    def __init__(self):
        super(MaskL1Loss, self).__init__()

    def forward(self, result, target, mask):
        l1 = torch.abs(result - target)
        masked_l1 = l1 * mask
        loss = masked_l1.sum() / mask.sum()
        return loss
