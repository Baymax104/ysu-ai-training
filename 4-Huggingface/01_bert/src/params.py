# -*- coding: UTF-8 -*-
import torch

LOCAL = False
DEBUG = False

# Loading
DEVICE = 'cuda' if torch.cuda.is_available() and not DEBUG else 'cpu'
NUM_WORKERS = 8 if not DEBUG else 0
PREFETCH_FACTOR = 3 if not DEBUG else None

# Train
MODEL_NAME = 'bert-base-uncased' if LOCAL else '../hub/bert-base-uncased'
EPOCHS = 20
BATCH_SIZE = 64
RANDOM_SEED = 200
LR = 1e-4
WEIGHT_DECAY = 1e-4

# System
DATA_DIR = '../data' if LOCAL else '/root/autodl-tmp/imdb'
LOG_DIR = '../logs' if LOCAL else '/root/tf-logs'
