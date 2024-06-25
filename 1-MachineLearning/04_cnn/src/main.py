# -*- coding: UTF-8 -*-
import os.path

import torch
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

import parameters as param
from cnn import CNN


def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_data = MNIST(root=param.DATA_DIR, train=True, transform=transform, download=True)
    test_data = MNIST(root=param.DATA_DIR, train=False, transform=transform, download=True)

    return DataLoader(train_data, batch_size=param.BATCH_SIZE, shuffle=True), \
        DataLoader(test_data, batch_size=param.BATCH_SIZE, shuffle=True)


def plot_metric(metric, label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f'CNN/{label}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(label)
    ax.plot(metric)
    plt.savefig(os.path.join(param.LOG_DIR, f'{label}.png'))


def train(model, train_loader):
    optimizer = SGD(model.parameters(), lr=param.LEARNING_RATE)
    criterion = CrossEntropyLoss().to(param.DEVICE)

    metrics = MetricCollection([
        MulticlassAccuracy(num_classes=10),
        MulticlassF1Score(num_classes=10)
    ]).to(param.DEVICE)

    loss_history = []
    acc_history = []
    f1_history = []

    model.train()
    for epoch in range(1, param.EPOCHS + 1):
        train_loss = torch.tensor(0.0).to(param.DEVICE)

        for data, target in tqdm(train_loader, desc=f'Training [Epoch: {epoch}/{param.EPOCHS}]'):
            data = data.to(param.DEVICE)
            target = target.to(param.DEVICE)

            optimizer.zero_grad()

            result = model(data)

            loss = criterion(result, target)
            train_loss += loss.item()

            metrics.update(result, target)

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_metrics = metrics.compute()
        metrics.reset()
        loss_history.append(train_loss.cpu())
        acc_history.append(train_metrics['MulticlassAccuracy'].cpu())
        f1_history.append(train_metrics['MulticlassF1Score'].cpu())

    print(f'Train [Loss]: {loss_history[-1]:.4f}')
    print(f'Train [Accuracy]: {acc_history[-1]:.4f}')
    print(f'Train [F1 Score]: {f1_history[-1]:.4f}')

    plot_metric(loss_history, 'Loss')
    plot_metric(acc_history, 'Accuracy')
    plot_metric(f1_history, 'F1-Score')


def test(model, test_loader):
    criterion = CrossEntropyLoss().to(param.DEVICE)

    metrics = MetricCollection([
        MulticlassAccuracy(num_classes=10),
        MulticlassF1Score(num_classes=10)
    ]).to(param.DEVICE)

    test_loss = torch.tensor(0.0).to(param.DEVICE)

    model.eval()
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f'Testing'):
            data, target = data.to(param.DEVICE), target.to(param.DEVICE)
            result = model(data)
            loss = criterion(result, target)
            test_loss += loss.item()
            metrics.update(result, target)

    test_loss /= len(test_loader)
    test_metrics = metrics.compute()
    metrics.reset()

    print(f'Test [Loss]: {test_loss:.4f}')
    print(f'Test [Accuracy]: {test_metrics["MulticlassAccuracy"]:.4f}')
    print(f'Test [F1-Score]: {test_metrics["MulticlassF1Score"]:.4f}')


if __name__ == '__main__':
    train_loader, test_loader = load_data()

    model = CNN().to(param.DEVICE)
    train(model, train_loader)
    test(model, test_loader)
