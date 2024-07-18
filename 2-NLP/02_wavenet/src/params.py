# -*- coding: UTF-8 -*-
import torch

LOCAL = True
DEBUG = True

# Loading
DEVICE = 'cuda' if torch.cuda.is_available() and not DEBUG else 'cpu'
NUM_WORKERS = 8 if not DEBUG else 0
PREFETCH = 3 if not DEBUG else 0

# Train
EPOCH = 50
BATCH_SIZE = 64
RANDOM_SEED = 404
LR = 0.001
WEIGHT_DECAY = 1e-4

# System
DATA_DIR = '../data' if LOCAL else '/root/autodl-tmp'
LOG_DIR = '../logs' if LOCAL else '/root/tf-logs'
