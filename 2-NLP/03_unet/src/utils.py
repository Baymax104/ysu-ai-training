# -*- coding: UTF-8 -*-
import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt

import params


def set_seed():
    torch.manual_seed(params.RANDOM_SEED)
    torch.cuda.manual_seed(params.RANDOM_SEED)
    random.seed(params.RANDOM_SEED)
    np.random.seed(params.RANDOM_SEED)


def plot_metrics(metrics, label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f'Wave-U-Net/{label}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(label)
    ax.plot(metrics)
    log_path = os.path.join(params.LOG_DIR, f'{label}.jpg')
    plt.savefig(log_path)
