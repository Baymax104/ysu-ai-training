# -*- coding: UTF-8 -*-
import torch
from prettytable import PrettyTable
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import params
from dataset import WaveDataset
from model.wavenet import WaveNet
from utils import plot_metrics, set_seed


def train(train_loader, model):
    model.train()

    optimizer = Adam(model.parameters(), lr=params.LR, weight_decay=params.WEIGHT_DECAY)
    criterion = CrossEntropyLoss().to(params.DEVICE)

    table = PrettyTable(field_names=['Epoch', 'Loss'])
    loss_history = []

    print('=' * 50 + f'Training in {params.DEVICE}' + '=' * 50)
    for e in range(1, params.EPOCH + 1):
        train_loss = 0

        for i, wav in enumerate(tqdm(train_loader, desc=f'Training Epoch [{e}/{params.EPOCH}]', colour='green')):
            wav = wav.to(params.DEVICE, non_blocking=True)
            result = model(wav)
            loss = criterion(result, wav) / params.ACCUM_STEP
            train_loss += loss.item()
            loss.backward()

            # gradient accumulation
            if (i + 1) % params.ACCUM_STEP == 0:
                optimizer.step()
                optimizer.zero_grad()

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
    criterion = CrossEntropyLoss().to(params.DEVICE)

    test_loss = 0
    for wav in tqdm(test_loader, desc=f'Testing', colour='green'):
        wav = wav.to(params.DEVICE)
        result = model(wav)
        loss = criterion(result, wav)
        test_loss += loss.item()
    test_loss /= len(test_loader)

    print(f'Testing [Loss]: {test_loss}')


if __name__ == '__main__':
    set_seed()
    train_data = WaveDataset(train=True)
    train_loader = DataLoader(train_data, batch_size=params.BATCH_SIZE, shuffle=True)

    model = WaveNet(in_channels=256, out_channels=256, k_layer=4, blocks=1).to(params.DEVICE)
    train(train_loader, model)
