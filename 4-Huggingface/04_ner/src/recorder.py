# -*- coding: UTF-8 -*-
from pathlib import Path
from typing import Any, Literal

import torch
from matplotlib import pyplot as plt
from prettytable import PrettyTable


class Recorder:

    def __init__(self, metrics: list[str], log_dir, steps=1, mode: Literal['train', 'test', 'validation'] = 'train'):
        assert steps > 0
        self.metrics = metrics
        self.mode = mode
        self.steps = steps
        self.log_dir = Path(log_dir)
        self.train = self.mode != 'test'
        field_names = (['Epoch'] if self.train else []) + self.metrics
        self.prettytable = PrettyTable(field_names=field_names)
        self.record_dict = {}
        for metric in self.metrics:
            self.record_dict[metric] = [0] * steps

    def add_record(self, step_dict, step=1):
        if step <= 0 or step > self.steps:
            raise IndexError(f'Step {step} is out of range[1, steps={self.steps}]')

        for metric, value in step_dict.items():
            if metric in self.record_dict:
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                self.record_dict[metric][step - 1] = value

        step_column = [step] if self.train else []
        row = step_column + [self.record_dict[metric][step - 1] for metric in self.metrics]
        self.prettytable.add_row(row)

    def print(self, clean=False):
        print(self.prettytable)
        if clean:
            self.prettytable.clear_rows()

    def plot(self, model_title='', *other_record):
        metrics = set(self.metrics).intersection(*[set(record.metrics) for record in other_record])
        records = [self] + [*other_record]
        max_steps = max([record.steps for record in records])
        for metric in metrics:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title(f'{model_title}/{metric}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            for record in records:
                # zoom steps
                if record.steps < max_steps:
                    ratio = max_steps // record.steps
                    steps = list(range(1, max_steps + 1))[::ratio][:record.steps]
                else:
                    steps = list(range(1, max_steps + 1))
                ax.plot(steps, record.record_dict[metric], label=record.mode)
            ax.legend()
            plt.savefig(self.log_dir / f'{metric}.jpg')

    def get_best(self, best_fn: list[str]) -> dict[str, Any]:
        best_fn = [fn for fn in best_fn if fn == 'max' or fn == 'min']
        best_dict = {}
        for metric, fn in zip(self.metrics, best_fn):
            values = self.record_dict[metric]
            if fn == 'max':
                best_value = max(values)
            else:
                best_value = min(values)
            best_dict[metric] = best_value
        return best_dict

    def keys(self) -> list[str]:
        return list(self.record_dict.keys())

    def items(self):
        for metric, value in self.record_dict.items():
            yield metric, value if self.train else value[0]

    def __getitem__(self, item):
        if item not in self.record_dict:
            raise KeyError(f'Key {item} is not in {self.metrics}')
        return self.record_dict[item] if self.train else self.record_dict[item][0]

    def __str__(self):
        return str(self.record_dict)
