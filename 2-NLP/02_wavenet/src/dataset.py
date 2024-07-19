# -*- coding: UTF-8 -*-
import os

import torch
import torchaudio
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchaudio.transforms import Resample

import params
from utils import mu_law


class WaveDataset(Dataset):

    def __init__(self, train=True):
        wav_dir = os.path.join(params.DATA_DIR, 'LJSpeech-1.1', 'wavs')
        assert os.path.isdir(wav_dir)
        files = os.listdir(wav_dir)

        if train:
            files = files[:len(files) * 8 // 10]  # 80%
        else:
            files = files[len(files) * 8 // 10:]  # 20%

        self.items = [os.path.join(wav_dir, wav_file) for wav_file in files]
        self.fix_wav_length = 223000

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path = self.items[idx]
        wav, sample_rate = torchaudio.load(wav_path)
        # fix time step
        if wav.size(dim=1) < self.fix_wav_length:
            padding = torch.zeros((1, self.fix_wav_length - wav.size(dim=1)))
            wav = torch.cat([wav, padding], dim=1)

        # down sampling
        resampler = Resample(sample_rate, 8000)
        wav = resampler(wav)

        # (channel=1, time_step)
        mu_law_code = mu_law(wav, mu=255.)
        # (time_step, 256)
        one_hot_code = one_hot(mu_law_code, num_classes=256).squeeze(dim=0).to(torch.float)
        return one_hot_code
