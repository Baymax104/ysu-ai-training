# -*- coding: UTF-8 -*-
import os

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

import params
from utils import to_mel


class Vocabulary:

    def __init__(self, metadata):
        all_txt = ''
        for txt in metadata['normalized']:
            all_txt += txt
        # get all characters
        self.chars = sorted(list(set(all_txt)))
        self.num_chars = len(self.chars)
        # char index map
        self.ctoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itoc = {i: ch for i, ch in enumerate(self.chars)}

        # padding character
        self.ctoi['_'] = self.num_chars
        self.itoc[self.num_chars] = '_'
        self.num_chars += 1

    def encode(self, text):
        return [self.ctoi[c] for c in text]

    def decode(self, indexes):
        return ''.join([self.itoc[i] for i in indexes])


class LJDataset(Dataset):

    def __init__(self, train):
        # (waveform, sample_rate, transcript, normalized_transcript)
        # raw_dataset = torchaudio.datasets.ljspeech.LJSPEECH(root=params.DATA_DIR, download=True)
        metadata_path = os.path.join(params.DATA_DIR, 'LJSpeech-1.1', 'metadata.csv')

        metadata = pd.read_csv(metadata_path, sep='|', header=None, names=['id', 'text', 'normalized'])
        metadata = metadata.dropna().reset_index(drop=True)
        for index, row in metadata.iterrows():
            _id = row['id']
            metadata.loc[index, 'wav_path'] = f'wavs/{_id}.wav'

        if train:
            self.metadata = metadata.iloc[:len(metadata) * 8 // 10]  # 80%
        else:
            self.metadata = metadata.iloc[len(metadata) * 8 // 10:]  # 20%

        self.vocab = Vocabulary(self.metadata)
        self.fix_text_length = 200
        self.fix_wav_length = 223000

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        metadata = self.metadata.iloc[idx]
        text = metadata['normalized']
        wav_path = metadata['wav_path']

        # processing text
        text_tensor = torch.tensor(self.vocab.encode(text), dtype=torch.long)
        # fix text length
        if len(text_tensor) < self.fix_text_length:
            padding = torch.ones(self.fix_text_length - len(text_tensor), dtype=torch.long) * self.vocab.num_chars
            text_tensor = torch.cat([text_tensor, padding])

        # processing wav
        wav_path = os.path.join(params.DATA_DIR, 'LJSpeech-1.1', wav_path)
        wav, sample_rate = torchaudio.load(wav_path)
        # fix wav length
        if wav.size(dim=1) < self.fix_wav_length:
            padding = torch.zeros((1, self.fix_wav_length - wav.size(dim=1)))
            wav = torch.cat([wav, padding], dim=1)
        mel_spectrogram = to_mel(wav, sample_rate)

        return text_tensor, mel_spectrogram
