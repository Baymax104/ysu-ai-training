# -*- coding: UTF-8 -*-
import os.path

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import parameters


class Recorder:

    def __init__(self):
        log_dir = os.path.join(parameters.LOG_DIR, datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_record(self, step, tag_value_dict):
        for tag, value in tag_value_dict.items():
            if type(value) is dict:
                self.writer.add_scalars(global_step=step, main_tag=tag, tag_scalar_dict=value)
            else:
                self.writer.add_scalar(global_step=step, tag=tag, scalar_value=value)

    def show(self):
        self.writer.close()
