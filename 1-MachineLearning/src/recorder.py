# -*- coding: UTF-8 -*-
import os.path

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import parameters


class Recorder:

    def __init__(self, main_tag):
        log_dir = os.path.join(parameters.LOG_DIR, datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.writer = SummaryWriter(log_dir=log_dir)
        self.main_tag = main_tag

    def add_record(self, step, tag_value_dict):
        for tag, value in tag_value_dict.items():
            self.writer.add_scalar(f'{self.main_tag}/{tag}', value, step)

    def show(self):
        self.writer.close()
