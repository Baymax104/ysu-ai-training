# -*- coding: UTF-8 -*-

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCH = 100
BATCH_SIZE = 64
