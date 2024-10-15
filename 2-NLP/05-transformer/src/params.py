# -*- coding: UTF-8 -*-
import torch

DEBUG = False
LOCAL = False

DEVICE = 'cuda' if torch.cuda.is_available() and not DEBUG else 'cpu'
IS_CPU = DEVICE == 'cpu'
NUM_WORKERS = 8 if not DEBUG else 0
PREFETCH_FACTOR = 3 if not DEBUG else None

TOKENIZER_NAME = 'google-bert/bert-base-uncased'
EPOCHS = 10
BATCH_SIZE = 8
RANDOM_SEED = 200
LR = 1e-5
WEIGHT_DECAY = 1e-4
MAX_LENGTH = 512
TRANSFORMER_PARAMETERS = {
    'n': 6,
    'd_model': 512,
    'd_ff': 2048,
    'h': 8,
    'dropout': 0.1
}

DATA_DIR = '../data' if LOCAL else '/root/autodl-tmp'
LOG_DIR = '../logs' if LOCAL else '/root/tf-logs'
