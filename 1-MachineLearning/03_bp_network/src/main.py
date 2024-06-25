# -*- coding: UTF-8 -*-
import os.path

import torch
from matplotlib import pyplot as plt
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from tqdm import tqdm

import parameters
from bp import BP, MSE


def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_data = MNIST(root=parameters.DATA_DIR, train=True, transform=transform, download=True)
    test_data = MNIST(root=parameters.DATA_DIR, train=False, transform=transform, download=True)

    return DataLoader(train_data, batch_size=len(train_data), shuffle=True), \
        DataLoader(test_data, batch_size=len(test_data), shuffle=True)


def plot(metric, label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f'BP/{label}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(label)
    ax.plot(metric)
    plt.savefig(os.path.join(parameters.LOG_DIR, f'{label}.png'))


def accuracy(y_pred, y_true):
    n = y_pred.size(0)
    pred_class = torch.argmax(y_pred, dim=1)
    true_class = torch.argmax(y_true, dim=1)
    acc = torch.sum(pred_class == true_class).float() / n
    return acc


def train(model, train_loader):
    data, target = next(iter(train_loader))
    data = data.view(data.size(0), -1)
    target = one_hot(target, num_classes=10)

    loss_history = []
    acc_history = []

    for _ in tqdm(range(1, parameters.EPOCHS + 1)):
        pred = model.fit(data, target)
        loss = MSE(pred, target)

        acc = accuracy(pred, target)
        loss_history.append(loss)
        acc_history.append(acc)

    print(f'Train [Loss]: {loss_history[-1]}')
    print(f'Train [Accuracy]: {acc_history[-1]}')

    plot(loss_history, 'Loss')
    plot(acc_history, 'Accuracy')


def test(model, test_loader):
    data, target = next(iter(test_loader))
    data = data.view(data.size(0), -1)
    target = one_hot(target, num_classes=10)
    pred = model.predict(data)
    loss = MSE(pred, target)
    acc = accuracy(pred, target)
    print(f'Test [Loss]: {loss}')
    print(f'Test [Accuracy]: {acc}')


if __name__ == '__main__':
    torch.manual_seed(parameters.RANDOM_SEED)
    train_loader, test_loader = load_data()
    bp = BP(lr=parameters.LEARNING_RATE)

    train(bp, train_loader)
    test(bp, test_loader)
