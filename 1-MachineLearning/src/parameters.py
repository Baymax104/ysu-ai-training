# -*- coding: UTF-8 -*-
import torch

EPOCH = 100
BATCH_SIZE = 2024
RANDOM_SEED = 22
OPTIM_PARAMS = {
    'MF': {
        'lr': 0.1,
        'weight_decay': 1e-4,
    },
    'GMF': {
        'lr': 0.01,
        'weight_decay': 1e-4,
    },
    'MLP': {
        'lr': 0.05,
        'weight_decay': 2e-4,
    }
}


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = '/root/autodl-tmp/dataset'  # for autodl
LOG_DIR = '/root/tf-logs'  # for autodl
# DEVICE = 'cpu'  # for debug
# DATA_DIR = '../dataset'  # for local
# LOG_DIR = '../logs'  # for local
