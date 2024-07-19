# -*- coding: UTF-8 -*-
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

import params


class EasyDataset(Dataset):
    r"""
    __getitem__(idx) -> dict:
        - input_ids(batch, max_length)
        - attention_mask(batch, max_length)
        - token_type_ids(batch, max_length)
        - label(batch, num_classes=40)
    """

    def __init__(self, train=True, max_length=256):
        self.max_length = max_length
        df_path = os.path.join(params.DATA_DIR, 'data.csv')
        label_path = os.path.join(params.DATA_DIR, 'label.npy')
        assert os.path.isfile(df_path) and os.path.isfile(label_path)

        df = pd.read_csv(df_path)  # (21000, 5)
        self.sentences = list(df['sentence'])
        labels = np.load(label_path)  # (21000, 40)
        self.labels = np.vsplit(labels, labels.shape[0])  # (1, 40) * 21000
        self.num_classes = self.labels[0].shape[1]
        assert len(self.sentences) == len(self.labels)

        self.sentences = self.sentences[:len(self.sentences) * 5 // 10]
        self.labels = self.labels[:len(self.labels) * 5 // 10]
        if train:
            self.sentences = self.sentences[:len(self.sentences) * 8 // 10]  # 80%
            self.labels = self.labels[:len(self.labels) * 8 // 10]
        else:
            self.sentences = self.sentences[len(self.sentences) * 8 // 10:]  # 20%
            self.labels = self.labels[len(self.labels) * 8 // 10:]

        self.tokenizer = BertTokenizer.from_pretrained(params.MODEL_NAME)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx].squeeze()  # (40,)
        encoding = self.tokenizer.encode_plus(
            sentence,
            padding='max_length',
            max_length=self.max_length,
            add_special_tokens=True,
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float)
        }
