# -*- coding: UTF-8 -*-
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import parameters


def preprocessing():
    rating_data = pd.read_csv(os.path.join(parameters.DATA_DIR, 'rating.csv'))

    # filter
    rating_data = rating_data[rating_data['rating'] != -1]

    # map to continuous index
    user_to_index = {user_id: i for i, user_id in enumerate(rating_data['user_id'].unique())}
    anime_to_index = {anime_id: i for i, anime_id in enumerate(rating_data['anime_id'].unique())}
    rating_data['user_id'] = rating_data['user_id'].map(user_to_index)
    rating_data['anime_id'] = rating_data['anime_id'].map(anime_to_index)

    user_num, anime_num = rating_data['user_id'].nunique(), rating_data['anime_id'].nunique()

    # split
    train_data, test_data = train_test_split(rating_data, test_size=0.2, random_state=parameters.RANDOM_SEED)
    train_data, validate_data = train_test_split(train_data, test_size=0.25, random_state=parameters.RANDOM_SEED)

    return train_data, test_data, validate_data, user_num, anime_num


class AnimeDataset(Dataset):

    def __init__(self, df):
        self.user_ids = torch.tensor(np.array(df['user_id']))
        self.anime_ids = torch.tensor(np.array(df['anime_id']))
        self.ratings = torch.tensor(np.array(df['rating']), dtype=torch.float)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.anime_ids[idx], self.ratings[idx]
