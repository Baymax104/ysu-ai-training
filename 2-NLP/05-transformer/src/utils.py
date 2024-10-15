# -*- coding: UTF-8 -*-
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_model(model, model_name, log_dir):
    model_dir = Path(log_dir) / 'model'
    if not model_dir.is_dir():
        model_dir.mkdir(exist_ok=True)
    model_name = f'{model_name}-{time.strftime("%Y%m%d-%H%M%S")}.pt'
    model_path = model_dir / model_name
    torch.save(model.state_dict(), model_path)


def load_model(model, model_file):
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
