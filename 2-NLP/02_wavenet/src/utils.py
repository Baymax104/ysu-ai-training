# -*- coding: UTF-8 -*-
import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt

import params


def mu_law(wav_tensor, mu=255.0):
    mu = torch.tensor(mu, dtype=torch.float32)
    y = torch.sign(wav_tensor) * torch.log(1.0 + mu * torch.abs(wav_tensor)) / torch.log(1.0 + mu)
    return torch.floor((y + 1) / 2 * mu).to(torch.int64)


def set_seed():
    torch.manual_seed(params.RANDOM_SEED)
    torch.cuda.manual_seed(params.RANDOM_SEED)
    random.seed(params.RANDOM_SEED)
    np.random.seed(params.RANDOM_SEED)


def plot_metrics(metrics, label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f'WavNet/{label}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(label)
    ax.plot(metrics)
    log_path = os.path.join(params.LOG_DIR, f'{label}.jpg')
    plt.savefig(log_path)
