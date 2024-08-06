# -*- coding: UTF-8 -*-
from pathlib import Path
from typing import Any

import torch
from matplotlib import pyplot as plt
from prettytable import PrettyTable


class Recorder:

    def __init__(self, metrics: list[str], steps=1):
        assert steps > 0
        self.metrics = metrics
        self.train = steps > 1
        field_names = (['Epoch'] if self.train else []) + self.metrics
        self.prettytable = PrettyTable(field_names=field_names)
        self.record_dict = {}
        self.steps = steps
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

    def plot(self, log_dir, model_title=''):
        for metric, values in self.record_dict.items():
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title(f'{model_title}/{metric}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.plot(values)
            log_path = Path(log_dir) / f'{metric}.jpg'
            plt.savefig(log_path)

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

    def __getitem__(self, item):
        if item not in self.record_dict:
            raise KeyError(f'Key {item} is not in {self.metrics}')
        return self.record_dict[item] if self.train else self.record_dict[item][0]

    def __str__(self):
        return str(self.record_dict)
