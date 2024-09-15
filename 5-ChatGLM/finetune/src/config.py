# -*- coding: UTF-8 -*-
import dataclasses as dc
from pathlib import Path
from typing import Union, Optional

from datasets import NamedSplit, Split
from peft import PeftConfig, get_peft_config
from transformers import Seq2SeqTrainingArguments, GenerationConfig

import utils


@dc.dataclass
class DataConfig(object):
    train_file: str
    val_file: Optional[str] = None
    test_file: Optional[str] = None

    num_proc: Optional[int] = None

    @property
    def data_format(self) -> str:
        return Path(self.train_file).suffix

    @property
    def data_files(self) -> dict[NamedSplit, str]:
        return {
            split: data_file
            for split, data_file in zip(
                [Split.TRAIN, Split.VALIDATION, Split.TEST],
                [self.train_file, self.val_file, self.test_file],
            ) if data_file is not None
        }


@dc.dataclass
class FinetuningConfig(object):
    data_config: DataConfig

    max_input_length: int
    max_output_length: int

    training_args: Seq2SeqTrainingArguments = dc.field(
        default_factory=lambda: Seq2SeqTrainingArguments(output_dir='./output')
    )
    peft_config: Optional[PeftConfig] = None

    def __post_init__(self):
        if not self.training_args.do_eval or self.data_config.val_file is None:
            # skips the evaluation stage when `do_eval` or `eval_file` is not provided
            self.training_args.do_eval = False
            self.training_args.evaluation_strategy = 'no'
            self.data_config.val_file = None
        else:
            self.training_args.per_device_eval_batch_size = self.training_args.per_device_eval_batch_size or self.training_args.per_device_train_batch_size

    @classmethod
    def from_dict(cls, **kwargs) -> 'FinetuningConfig':
        training_args: dict | None = kwargs.get('training_args', None)
        if training_args is not None and not isinstance(training_args, Seq2SeqTrainingArguments):
            gen_config: dict | None = training_args.get('generation_config')
            # a bit hacky
            if not isinstance(gen_config, GenerationConfig):
                training_args['generation_config'] = GenerationConfig(**gen_config)
            kwargs['training_args'] = Seq2SeqTrainingArguments(**training_args)

        data_config: dict | None = kwargs.get('data_config')
        if not isinstance(data_config, DataConfig):
            kwargs['data_config'] = DataConfig(**data_config)

        peft_config: dict | None = kwargs.get('peft_config', None)
        if peft_config is not None and not isinstance(peft_config, PeftConfig):
            kwargs['peft_config'] = get_peft_config(peft_config)
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'FinetuningConfig':
        path = utils.resolve_path(path)
        kwargs = utils.get_yaml_parser().load(path)
        return cls.from_dict(**kwargs)
