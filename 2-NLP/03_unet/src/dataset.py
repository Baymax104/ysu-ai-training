# -*- coding: UTF-8 -*-
import torch
from torch.utils.data import Dataset
from torchaudio.datasets.librimix import LibriMix

import params


class MixtureDataset(Dataset):
    r"""
    Item(mixture, source1, source2)
        - mixture(batch, channel=1, time_step)
        - source1(batch, channel=1, time_step)
        - source2(batch, channel=1, time_step)
    """

    def __init__(self, train=True):
        # (sample_rate, mixture, [source1, source2])
        # mixture(1, time_step)
        raw_dataset = LibriMix(root=params.DATA_DIR, subset='dev')
        length = len(raw_dataset)
        r = range(0, length * 8 // 10) if train else range(length * 8 // 10, length)
        self.mixtures = [raw_dataset[i][1] for i in r]
        self.sources1 = [raw_dataset[i][2][0] for i in r]
        self.sources2 = [raw_dataset[i][2][1] for i in r]

        # (max(wav_lengths) + min(wav_lengths)) // 2 = 81880
        # approximately equal to 2^20 = 81920, fix length to 81920
        self.fix_length = 81920

        assert len(self.mixtures) == len(self.sources1) == len(self.sources2)

    def __len__(self):
        return len(self.mixtures)

    def __getitem__(self, idx):
        mixture = self.mixtures[idx]
        source1 = self.sources1[idx]
        source2 = self.sources2[idx]

        assert mixture.size() == source1.size() == source2.size()

        # truncation or padding
        if mixture.size(dim=1) > self.fix_length:
            mixture = mixture[:, :self.fix_length]
            source1 = source1[:, :self.fix_length]
            source2 = source2[:, :self.fix_length]
        elif mixture.size(dim=1) < self.fix_length:
            blank = torch.zeros((mixture.size(dim=0), self.fix_length - mixture.size(dim=1)), dtype=mixture.dtype)
            mixture = torch.cat([mixture, blank], dim=1)
            source1 = torch.cat([source1, blank], dim=1)
            source2 = torch.cat([source2, blank], dim=1)

        return mixture, (source1, source2)
