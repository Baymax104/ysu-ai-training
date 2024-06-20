# -*- coding: UTF-8 -*-
import torch

EPOCH = 100
BATCH_SIZE = 2024
LEARNING_RATE = 0.1
WEIGHT_DECAY = 1e-4


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = '/root/autodl-tmp/dataset'  # for autodl
LOG_DIR = '/root/tf-logs'  # for autodl
# DEVICE = 'cpu'  # for debug
# DATA_DIR = '../dataset'  # for local
# LOG_DIR = '../logs'  # for local
RANDOM_SEED = 22
