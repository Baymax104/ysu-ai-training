# -*- coding: UTF-8 -*-
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import params


class CoNLLDataset(Dataset):

    def __init__(self, train=True):
        name = 'train' if train else 'test'
        data_file = str(Path(params.DATA_DIR) / f'conll2003-{name}.arrow')
        self.dataset = load_dataset('arrow', data_files=data_file)['train']

    def __getitem__(self, idx):
        return self.dataset[idx]['tokens'], self.dataset[idx]['ner_tags']

    def __len__(self):
        return self.dataset.num_rows


class Collator:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(params.MODEL_NAME)

    def __call__(self, batch):
        batch_tokens, batch_ner_tags = zip(*batch)

        # encode tokens
        encodings = self.tokenizer.batch_encode_plus(
            list(batch_tokens),
            is_split_into_words=True,
            padding='max_length',
            max_length=256,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )

        # align labels
        batch_aligned_labels = []
        for i, ner_tags in enumerate(batch_ner_tags):
            word_ids = encodings.word_ids(batch_index=i)
            token_labels = []
            for j in range(len(word_ids)):
                word_id = word_ids[j]
                if word_id is None:
                    token_labels.append(0)
                elif j > 0 and word_ids[j - 1] == word_id:
                    token_labels.append(0)
                else:
                    token_labels.append(ner_tags[word_id])
            batch_aligned_labels.append(token_labels)
        batch_aligned_labels = torch.tensor(batch_aligned_labels)

        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': batch_aligned_labels
        }
