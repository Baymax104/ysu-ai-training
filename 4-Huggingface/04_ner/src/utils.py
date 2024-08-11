# -*- coding: UTF-8 -*-
import time
from pathlib import Path

import numpy as np
import torch

import params


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_model(model, model_name):
    model_dir = Path(params.LOG_DIR) / 'model'
    if not model_dir.is_dir():
        model_dir.mkdir(exist_ok=True)
    model_name = f'{model_name}-{time.strftime("%Y%m%d-%H%M%S")}.pt'
    model_path = model_dir / model_name
    torch.save(model.state_dict(), model_path)


def load_model(model, model_filename):
    model_dir = Path(params.LOG_DIR) / 'model'
    if not model_dir.is_dir():
        raise ValueError(f'Model directory {model_dir} does not exist.')
    model_path = model_dir / model_filename
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
