# -*- coding: UTF-8 -*-
import torch
from prettytable import PrettyTable
from torch.optim import RAdam
from torch.utils.data import DataLoader
from tqdm import tqdm

import params
from dataset import LJDataset
from model.loss import MaskL1Loss
from model.tacotron import Tacotron
from utils import set_seed, plot_metrics, align


def train(train_loader, model):
    model.train()

    criterion = MaskL1Loss().to(params.DEVICE)
    optimizer = RAdam(model.parameters(), lr=params.LR, weight_decay=params.WEIGHT_DECAY)
    table = PrettyTable(field_names=['Epoch', 'Loss'])
    loss_history = []

    print('=' * 40 + f'Training in {params.DEVICE}' + '=' * 40)
    for e in range(1, params.EPOCH + 1):
        train_loss = 0

        for text, mel in tqdm(train_loader, desc=f'Training Epoch [{e}/{params.EPOCH}]', colour='green'):
            text, mel = text.to(params.DEVICE), mel.to(params.DEVICE)

            optimizer.zero_grad()
            result = model(text)
            result, mask = align(result, mel, 1)
            loss = criterion(result, mel, mask)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        loss_history.append(train_loss)
        table.add_row([e, train_loss])
        if e % 10 == 0:
            print(table)
            table.clear_rows()

    print(f'Training [Loss]: {min(loss_history)}')
    plot_metrics(loss_history, 'Loss')


@torch.no_grad()
def test(test_loader, model):
    model.eval()

    criterion = MaskL1Loss().to(params.DEVICE)
    test_loss = 0
    for text, mel in tqdm(test_loader, desc=f'Testing', colour='green'):
        text, mel = text.to(params.DEVICE, non_blocking=True), mel.to(params.DEVICE, non_blocking=True)
        result = model(text)
        result, mask = align(result, mel, 1)
        loss = criterion(result, mel, mask)
        test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f'Training [Loss]: {test_loss}')


if __name__ == '__main__':
    set_seed()
    train_data = LJDataset(train=True)
    test_data = LJDataset(train=False)
    train_loader = DataLoader(train_data, batch_size=params.BATCH_SIZE, shuffle=True, num_workers=params.NUM_WORKERS,
                              prefetch_factor=params.PREFETCH, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=params.BATCH_SIZE, num_workers=params.NUM_WORKERS,
                             prefetch_factor=params.PREFETCH, pin_memory=True)

    model = Tacotron(num_chars=200).to(params.DEVICE)
    train(train_loader, model)
