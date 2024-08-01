# -*- coding: UTF-8 -*-
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import params


class ImdbDataset(Dataset):

    def __init__(self, train=True):
        # (review, sentiment) * 50000
        data: pd.DataFrame = pd.read_csv(Path(params.DATA_DIR) / 'data.csv')
        if train:
            data, _ = train_test_split(data, test_size=0.2, random_state=params.RANDOM_SEED)
        else:
            _, data = train_test_split(data, test_size=0.2, random_state=params.RANDOM_SEED)
        data.reset_index(inplace=True, drop=True)
        data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
        self.items = data

        self.tokenizer = AutoTokenizer.from_pretrained(params.MODEL_NAME)
        self.max_length = 256

    def __getitem__(self, index):
        text, label = self.items.loc[index]
        label = torch.tensor(label)
        label = one_hot(label, num_classes=2).to(torch.float)
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze(),
            'label': label
        }

    def __len__(self):
        return len(self.items)
