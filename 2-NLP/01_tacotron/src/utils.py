# -*- coding: UTF-8 -*-
import os.path
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchaudio.transforms import MelSpectrogram

import params


def to_mel(waveform, sample_rate):
    mel_transform = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=80,
    )
    mel_spectrogram = mel_transform(waveform)
    mel_spectrogram = torch.log(mel_spectrogram + 1e-9)
    mel_spectrogram = mel_spectrogram.squeeze(dim=0).transpose(0, 1)
    return mel_spectrogram


def align(data: torch.Tensor, target: torch.Tensor, dim: int):
    assert len(data.size()) == len(target.size())
    size_length = len(data.size())
    size = []
    for d in range(size_length):
        s = target.size(dim=d) - data.size(dim=d) if d == dim else data.size(dim=d)
        size.append(s)
    blank = torch.zeros(size, device=data.device, dtype=data.dtype)
    mask = torch.ones(data.size(), device=data.device, dtype=data.dtype)
    mask = torch.cat([mask, blank], dim=dim)
    alignment = torch.cat([data, blank], dim=dim)
    return alignment, mask


def set_seed():
    torch.manual_seed(params.RANDOM_SEED)
    torch.cuda.manual_seed(params.RANDOM_SEED)
    random.seed(params.RANDOM_SEED)
    np.random.seed(params.RANDOM_SEED)


def plot_metrics(metrics, label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f'Tacotron/{label}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(label)
    ax.plot(metrics)
    log_path = os.path.join(params.LOG_DIR, f'{label}.jpg')
    plt.savefig(log_path)
