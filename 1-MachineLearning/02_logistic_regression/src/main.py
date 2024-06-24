# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import log_loss, accuracy_score, f1_score
from prettytable import PrettyTable
from tqdm import tqdm
from matplotlib import pyplot as plt

from LogisticRegression import LogisticRegression

random_seed = 500
epoch = 100
learning_rate = 0.01


def load_data():
    # iris_data = load_iris()
    x_data = pd.read_csv('../dataset/iris_data.csv')
    y_data = pd.read_csv('../dataset/iris_target.csv')
    y_label = pd.read_csv('../dataset/iris_label.csv')

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=random_seed)
    scaler = StandardScaler()
    scaler.fit_transform(x_train)
    scaler.transform(x_test)

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = np.array(x_test), np.array(y_test)
    return x_train, x_test, y_train, y_test, y_label


def plot_metrics(metric, label, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.title.set_text(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(label)
    ax.plot(metric)
    plt.savefig(f'../logs/{label}.png')


def train(model, x_train, y_train):
    table = PrettyTable()
    table.field_names = ['Epoch', 'Loss', 'accuracy', 'f1']
    loss_history = []
    acc_history = []
    f1_score_history = []

    for e in tqdm(range(1, epoch + 1)):
        predication = model.fit(x_train, y_train)
        loss = log_loss(y_train, predication)
        # transform to class
        predication = np.round(predication)

        acc = accuracy_score(y_train, predication)
        f1 = f1_score(y_train, predication)
        loss_history.append(loss)
        acc_history.append(acc)
        f1_score_history.append(f1)

        table.add_row([e, loss, acc, f1])
    print(table)
    print(f'Train [Ent]: {loss_history[-1]}')
    print(f'Train [Accuracy]: {acc_history[-1]}')
    print(f'Train [F1 Score]: {f1_score_history[-1]}')

    # plot
    plot_metrics(loss_history, 'Loss', 'Train/Loss')
    plot_metrics(acc_history, 'Accuracy', 'Train/Accuracy')
    plot_metrics(f1_score_history, 'F1-Score', 'Train/F1 Score')


def test(model, x_test, y_test):
    prediction = model.predict(x_test)
    prediction = np.round(prediction)
    loss = log_loss(y_test, prediction)
    acc = accuracy_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    print(f'Test [Ent]: {loss}')
    print(f'Test [Accuracy]: {acc}')
    print(f'Test [F1 Score]: {f1}')


if __name__ == '__main__':
    x_train, x_test, y_train, y_test, y_label = load_data()
    model = LogisticRegression(x_train.shape[1], x_train.shape[0], learning_rate)
    train(model, x_train, y_train)
    test(model, x_test, y_test)
