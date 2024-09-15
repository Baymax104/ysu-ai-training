# -*- coding: UTF-8 -*-
from pathlib import Path
from typing import Optional, Callable, Any

import numpy as np
from datasets import NamedSplit, DatasetDict, load_dataset, Dataset
from transformers import DataCollatorForSeq2Seq

import utils
from config import DataConfig


def load_datasets(
    data_dir: Path,
    data_format: str,
    data_files: dict[NamedSplit, str],
    num_proc: Optional[int]
) -> DatasetDict:
    if data_format in ('.csv', '.json', '.jsonl'):
        dataset_dct = load_dataset(data_format[1:], data_dir=str(data_dir), data_files=data_files, num_proc=num_proc)
    else:
        err_msg = f"Cannot load dataset in the '{data_format}' format."
        raise NotImplementedError(err_msg)

    return dataset_dct


def process_batch(batch, tokenizer, max_input_length, max_output_length) -> dict[str, list]:
    batched_tools = batch.get('tools', None)
    batched_conv = batch['conversations']
    batched_input_ids = []
    batched_labels = []

    if batched_tools is None:
        batched_tools = [None] * len(batched_conv)

    for tools, conv in zip(batched_tools, batched_conv):
        input_ids = [tokenizer.get_command('[gMASK]'), tokenizer.get_command('sop')]
        loss_masks = [False, False]

        if tools is not None:
            raise NotImplementedError()

        for message in conv:
            if message['role'] == 'tool':
                raise NotImplementedError()

            loss_mask_val = message['role'] not in ('system', 'user')
            new_input_ids = tokenizer.build_single_message(message['role'], '', message['content'])
            new_loss_masks = [loss_mask_val] * len(new_input_ids)

            input_ids += new_input_ids
            loss_masks += new_loss_masks

        input_ids.append(tokenizer.eos_token_id)
        loss_masks = [False, *loss_masks]
        labels = []
        for input_id, mask in zip(input_ids, loss_masks):
            labels.append(input_id if mask else -100)

        max_length = max_input_length + max_output_length + 1
        batched_input_ids.append(input_ids[:max_length])
        batched_labels.append(labels[:max_length])
    return {'input_ids': batched_input_ids, 'labels': batched_labels}


def process_batch_eval(batch, tokenizer, max_input_length, max_output_length) -> dict[str, list]:
    batched_tools = batch.get('tools', None)
    batched_conv = batch['conversations']
    batched_input_ids = []
    # To avoid computing loss, we do not provide the `labels` field in the input dictionary.
    batched_output_ids = []

    if batched_tools is None:
        batched_tools = [None] * len(batched_conv)

    for tools, conv in zip(batched_tools, batched_conv):
        input_ids = [tokenizer.get_command('[gMASK]'), tokenizer.get_command('sop')]

        if tools is not None:
            raise NotImplementedError()

        for message in conv:
            if len(input_ids) >= max_input_length:
                break
            if message['role'] == 'tool':
                raise NotImplementedError()

            new_input_ids = tokenizer.build_single_message(message['role'], '', message['content'])
            if message['role'] == 'assistant':
                output_prompt, output_ids = (new_input_ids[:1], new_input_ids[1:])
                output_ids.append(tokenizer.eos_token_id)
                batched_input_ids.append(input_ids[:max_input_length] + output_prompt[:1])
                batched_output_ids.append(output_ids[:max_output_length])
            input_ids += new_input_ids

    return {'input_ids': batched_input_ids, 'output_ids': batched_output_ids}


class GLMDataCollator(DataCollatorForSeq2Seq):

    def __call__(self, features, return_tensors=None):
        output_ids = ([feature['output_ids'] for feature in features] if 'output_ids' in features[0].keys() else None)
        if output_ids is not None:
            max_output_length = max(len(out) for out in output_ids)
            if self.pad_to_multiple_of is not None:
                max_output_length += (self.pad_to_multiple_of - 1)
                max_output_length = max_output_length // self.pad_to_multiple_of * self.pad_to_multiple_of
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (max_output_length - len(feature['output_ids']))
                if isinstance(feature['output_ids'], list):
                    feature['output_ids'] = feature['output_ids'] + remainder
                else:
                    feature['output_ids'] = np.concatenate([feature['output_ids'], remainder]).astype(np.int64)

        return super().__call__(features, return_tensors)


class DataManager(object):
    def __init__(self, data_dir: str, data_config: DataConfig):
        self._num_proc = data_config.num_proc

        self._dataset_dct = load_datasets(
            data_dir=utils.resolve_path(data_dir),
            data_format=data_config.data_format,
            data_files=data_config.data_files,
            num_proc=self._num_proc,
        )

    def get_dataset(self,
                    split: NamedSplit,
                    process_fn: Callable[[dict[str, Any]], dict[str, Any]],
                    batched: bool = True,
                    remove_orig_columns: bool = True) -> Optional[Dataset]:
        orig_dataset: Dataset | None = self._dataset_dct.get(split, None)
        if orig_dataset is None:
            return
        remove_columns = orig_dataset.column_names if remove_orig_columns else None

        return orig_dataset.map(
            process_fn,
            batched=batched,
            remove_columns=remove_columns,
            num_proc=self._num_proc,
        )
