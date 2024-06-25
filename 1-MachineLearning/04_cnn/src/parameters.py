# -*- coding: UTF-8 -*-
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.01

DATA_DIR = '../dataset'  # for local
# DATA_DIR = '/root/autodl-tmp'  # for autodl
LOG_DIR = '../logs'
