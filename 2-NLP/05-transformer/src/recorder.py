# -*- coding: UTF-8 -*-
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from prettytable import PrettyTable


class Recorder:

    def __init__(self, metrics: list[str], steps=1, log_dir='./logs', use_tensorboard=False):
        assert steps > 0
        self.metrics = metrics
        self.train = steps > 1
        self.log_dir = Path(log_dir)
        self.use_tensorboard = use_tensorboard
        field_names = (['Epoch'] if self.train else []) + metrics
        self.prettytable = PrettyTable(field_names=field_names)
        self.record_dict = {}
        self.steps = steps

        for metric in metrics:
            self.record_dict[metric] = [0] * steps

        self.log_dir.mkdir(exist_ok=True)

        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.summary_writer = SummaryWriter(str(self.log_dir))

    def add_record(self, step_dict, step=1):
        if step <= 0 or step > self.steps:
            raise IndexError(f'Step {step} is out of range[1, steps={self.steps}]')

        for metric, value in step_dict.items():
            if metric in self.record_dict:
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                self.record_dict[metric][step - 1] = value

                if self.use_tensorboard:
                    self.summary_writer.add_scalar(metric, value, step)

        step_column = [step] if self.train else []
        row = step_column + [self.record_dict[metric][step - 1] for metric in self.metrics]
        self.prettytable.add_row(row)

    def print(self, clean=False):
        print(self.prettytable)
        if clean:
            self.prettytable.clear_rows()

    def plot(self, model_title=''):
        for metric, values in self.record_dict.items():
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title(f'{model_title}/{metric}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.plot(values)
            log_path = self.log_dir / f'{metric}.jpg'
            plt.savefig(log_path)

    def __getitem__(self, item):
        if item not in self.record_dict:
            raise KeyError(f'Key {item} is not in {self.metrics}')
        return self.record_dict[item] if self.train else self.record_dict[item][0]
