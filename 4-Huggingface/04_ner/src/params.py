# -*- coding: UTF-8 -*-
import torch

LOCAL = False
DEBUG = False

# Loading
DEVICE = 'cuda' if torch.cuda.is_available() and not DEBUG else 'cpu'
NUM_WORKERS = 8 if not DEBUG else 0
PREFETCH_FACTOR = 3 if not DEBUG else None

# Train
MODEL_NAME = 'google-bert/bert-base-uncased'
EPOCHS = 15
BATCH_SIZE = 32
RANDOM_SEED = 200
LR = 2e-5
LR_STEPS = 5
LR_RATIO = 0.1
WEIGHT_DECAY = 1e-3

# System
DATA_DIR = '../data' if LOCAL else '/root/autodl-tmp/conll2003'
LOG_DIR = '../logs' if LOCAL else '/root/tf-logs'
