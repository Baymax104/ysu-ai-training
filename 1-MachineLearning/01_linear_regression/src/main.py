# -*- coding: UTF-8 -*-
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from prettytable import PrettyTable

from LinearRegression import LinearRegression

random_seed = 500
learning_rate = 0.05
epoch = 300


def load_data():
    # house_data = fetch_california_housing()
    x_data = pd.read_csv('../dataset/house_data.csv')
    y_data = pd.read_csv('../dataset/house_target.csv')

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=random_seed)
    scalar = StandardScaler()
    x_train = scalar.fit_transform(x_train)
    x_test = scalar.transform(x_test)
    x_train, x_test = np.array(x_train), np.array(x_test)
    y_train, y_test = np.array(y_train), np.array(y_test)
    return x_train, x_test, y_train, y_test


def plot_loss(loss_history):
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.plot(loss_history)
    plt.savefig('../logs/loss.png')


def train(model, x_train, y_train):
    table = PrettyTable()
    table.field_names = ['Epoch', 'MSE']
    loss_history = []

    # train
    for e in tqdm(range(1, epoch + 1)):
        y_hat = model.fit(x_train, y_train)
        loss = mean_squared_error(y_train, y_hat)
        loss_history.append(loss)
        table.add_row([e, loss])
    print(table)
    print(f'Train [MSE]: {loss_history[-1]}')

    plot_loss(loss_history)


def test(model, x_test, y_test):
    prediction = model.predict(x_test)
    loss = mean_squared_error(y_test, prediction)
    print(f'Test [MSE]: {loss}')


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    feature_dim = x_train.shape[1]
    sample_num = x_train.shape[0]

    model = LinearRegression(feature_dim, sample_num, learning_rate)

    train(model, x_train, y_train)
    test(model, x_test, y_test)
