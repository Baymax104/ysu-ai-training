# -*- coding: UTF-8 -*-
import torch
from prettytable import PrettyTable
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import params
from dataset import EasyDataset
from model import BertMultiClassification
from utils import set_seed, plot_metrics, accuracy


def train(train_loader, model):
    model.train()

    criterion = BCEWithLogitsLoss().to(params.DEVICE)
    optimizer = Adam(model.parameters(), lr=params.LR, weight_decay=params.WEIGHT_DECAY)

    table = PrettyTable(field_names=['Epoch', 'Loss', 'Accuracy'])

    print('=' * 50 + f'Training in {params.DEVICE}' + '=' * 50)
    loss_history = []
    acc_history = []
    for e in range(1, params.EPOCH + 1):
        train_loss = 0
        train_acc = 0
        for i, sample in enumerate(tqdm(train_loader, desc=f'Training Epoch [{e}/{params.EPOCH}]', colour='green')):
            input_ids = sample['input_ids'].to(params.DEVICE)
            attention_mask = sample['attention_mask'].to(params.DEVICE)
            token_type_ids = sample['token_type_ids'].to(params.DEVICE)
            label = sample['label'].to(params.DEVICE)

            optimizer.zero_grad()
            result = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(result, label)
            acc = accuracy(result, label)
            train_loss += loss.item()
            train_acc += acc

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        loss_history.append(train_loss)
        acc_history.append(train_acc)
        table.add_row([e, train_loss, train_acc])

        if e % 5 == 0:
            print(table)
            table.clear_rows()

    print(f'Training [Loss]: {min(loss_history)}')
    print(f'Training [Accuracy]: {max(acc_history)}')
    plot_metrics(loss_history, 'Loss')
    plot_metrics(acc_history, 'Accuracy')


@torch.no_grad()
def test(test_loader, model):
    model.eval()
    criterion = BCEWithLogitsLoss().to(params.DEVICE)
    # acc_metric = MulticlassAccuracy(num_classes=model.num_classes).to(params.DEVICE)

    test_loss = 0
    for sample in tqdm(test_loader, desc=f'Testing in {params.DEVICE}', colour='green'):
        input_ids = sample['input_ids'].to(params.DEVICE)
        attention_mask = sample['attention_mask'].to(params.DEVICE)
        token_type_ids = sample['token_type_ids'].to(params.DEVICE)
        label = sample['label'].to(params.DEVICE)
        result = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(result, label)
        test_loss += loss.item()
    test_loss /= len(test_loader)

    print(f'Testing [Loss]: {test_loss}')
    # print(f'Testing [Accuracy]: {acc}')


if __name__ == '__main__':
    set_seed()
    train_data = EasyDataset(train=True)
    test_data = EasyDataset(train=False)
    train_loader = DataLoader(train_data, batch_size=params.BATCH_SIZE, shuffle=True, num_workers=params.NUM_WORKERS,
                              prefetch_factor=params.PREFETCH_FACTOR, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=params.BATCH_SIZE, shuffle=False, num_workers=params.NUM_WORKERS)

    model = BertMultiClassification(num_classes=train_data.num_classes).to(params.DEVICE)
    train(train_loader, model)
