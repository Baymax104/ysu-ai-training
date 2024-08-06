# -*- coding: UTF-8 -*-
from pathlib import Path

import torch
from datasets import load_dataset
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import params


class SST2Dataset(Dataset):

    def __init__(self, train=True):
        name = 'train' if train else 'test'
        self.dataset = load_dataset('arrow', data_files=str(Path(params.DATA_DIR) / f'sst2-{name}.arrow'))['train']
        self.tokenizer = AutoTokenizer.from_pretrained(params.MODEL_NAME)

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, idx):
        return self.dataset[idx]['text'], self.dataset[idx]['label']


class Collator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        batch_sentences, batch_labels = zip(*batch)

        # input_ids(batch, 256)
        # attention_mask(batch, 256)
        batch_encoding = self.tokenizer.batch_encode_plus(
            batch_sentences,
            padding='max_length',
            max_length=256,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # (batch, 2)
        batch_labels = torch.tensor(batch_labels)
        batch_labels = one_hot(batch_labels, num_classes=2).to(torch.float)

        return {
            'input_ids': batch_encoding['input_ids'],
            'attention_mask': batch_encoding['attention_mask'],
            'labels': batch_labels,
        }
