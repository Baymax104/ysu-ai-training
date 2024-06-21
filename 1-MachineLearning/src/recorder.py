# -*- coding: UTF-8 -*-
import os.path
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

import parameters


class Recorder:

    def __init__(self, model_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(parameters.LOG_DIR, f'{model_name}_{timestamp}')
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_train_record(self, step, metric_dict):
        for tag, value in metric_dict.items():
            self.writer.add_scalar(global_step=step, tag=f'{tag}/Train', scalar_value=value)

    def add_validate_record(self, step, metric_dict):
        for tag, value in metric_dict.items():
            self.writer.add_scalar(global_step=step, tag=f'{tag}/Validate', scalar_value=value)

    def show(self):
        self.writer.close()
