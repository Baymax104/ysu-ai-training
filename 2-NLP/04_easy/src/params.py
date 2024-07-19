# -*- coding: UTF-8 -*-
import torch

LOCAL = False
DEBUG = False
MODEL_NAME = 'bert-base-uncased' if LOCAL else '../hub/bert-base-uncased'

# Loading
DEVICE = 'cuda' if torch.cuda.is_available() and not DEBUG else 'cpu'
NUM_WORKERS = 8 if not DEBUG else 0
PREFETCH_FACTOR = 3 if not DEBUG else None

# Train
EPOCH = 30
BATCH_SIZE = 32
ACCUMULATION_STEP = 4
RANDOM_SEED = 404
LR = 1e-4
WEIGHT_DECAY = 1e-4

# System
DATA_DIR = '../data' if LOCAL else '/root/autodl-tmp/easy'
LOG_DIR = '../logs' if LOCAL else '/root/tf-logs'
