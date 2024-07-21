# -*- coding: UTF-8 -*-
import os
import random
import time

import numpy as np
import torch
from transformers import set_seed as transformers_set_seed

import params


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    transformers_set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_model(model, model_name):
    model_dir = os.path.join(params.LOG_DIR, 'model')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    model_name = f'{model_name}-{time.strftime("%Y%m%d-%H%M%S")}.pt'
    model_path = os.path.join(model_dir, model_name)
    torch.save(model.state_dict(), model_path)


def load_model(model, model_filename):
    model_dir = os.path.join(params.LOG_DIR, 'model')
    if not os.path.isdir(model_dir):
        raise ValueError(f'Model directory {model_dir} does not exist.')
    model_path = os.path.join(model_dir, model_filename)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
