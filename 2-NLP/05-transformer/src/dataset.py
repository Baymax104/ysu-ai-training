# -*- coding: UTF-8 -*-
from typing import Literal

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm


class WMTDataset(Dataset):

    def __init__(self, split: Literal['train', 'validation', 'test']):
        data = load_dataset('wmt/wmt17', 'zh-en', split=split)

        if split == 'train':
            data = data[:2]['translation']
        else:
            data = data[:]['translation']

        self.items = []
        for sample in tqdm(data, desc=f'Loading {split} data', total=len(data)):
            en = sample['en']
            zh = sample['zh']
            self.items.append((en, zh))

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)


class Collator:

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        batch_input, batch_output = zip(*batch)

        batch_input = self.tokenizer.batch_encode_plus(
            list(batch_input),
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )

        batch_output = self.tokenizer.batch_encode_plus(
            list(batch_output),
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )

        causal_mask = torch.tril(torch.ones(self.max_length, self.max_length, dtype=torch.float)).unsqueeze(0)

        return {
            'input_ids': batch_input['input_ids'],
            'input_attention_mask': batch_input['attention_mask'],
            'output_ids': batch_output['input_ids'],
            'output_attention_mask': batch_output['attention_mask'],
            'causal_mask': causal_mask
        }
