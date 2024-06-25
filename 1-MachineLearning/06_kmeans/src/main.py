# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from kmeans import KMeans

RANDOM_SEED = 100
K = 3
EPOCH = 30


def load_data():
    data = pd.read_csv('../dataset/iris_data.csv')
    target = pd.read_csv('../dataset/iris_target.csv')
    data, target = np.array(data), np.array(target).squeeze()
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2,
                                                                        random_state=RANDOM_SEED)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    return train_data, test_data, train_target, test_target


def accuracy(y_pred, y_true):
    n = y_true.shape[0]
    return np.sum(y_pred == y_true) / n


def train(model, train_data, train_target):
    acc_history = []
    for _ in tqdm(range(EPOCH)):
        closest_class = model.fit(train_data)
        acc = accuracy(closest_class, train_target)
        acc_history.append(acc)

    print(f'Train [Accuracy]: {max(acc_history)}')


def test(model, test_data, test_target):
    closest_class = model.predict(test_data)
    acc = accuracy(closest_class, test_target)
    print(f'Test [Accuracy]: {acc}')


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    train_data, test_data, train_target, test_target = load_data()
    kmeans = KMeans(K, train_data)

    train(kmeans, train_data, train_target)
    test(kmeans, test_data, test_target)
