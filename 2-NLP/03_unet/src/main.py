# -*- coding: UTF-8 -*-
import torch
from prettytable import PrettyTable
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import params
from dataset import MixtureDataset
from model.waveunet import WaveUNet
from utils import plot_metrics, save_model, load_model, set_seed


def train(train_loader, model):
    model.train()

    criterion = MSELoss().to(params.DEVICE)
    optimizer = Adam(model.parameters(), lr=params.LR, weight_decay=params.WEIGHT_DECAY)

    loss_history = []
    table = PrettyTable(field_names=['Epoch', 'Loss'])

    print('=' * 50 + f'Training in {params.DEVICE}' + '=' * 50)
    for e in range(1, params.EPOCH + 1):
        train_loss = 0.

        for mix, srcs in tqdm(train_loader, desc=f'Training Epoch [{e}/{params.EPOCH}]', colour='green'):
            mix = mix.to(params.DEVICE)
            srcs = torch.stack(srcs, dim=0).to(params.DEVICE)

            optimizer.zero_grad()
            results = model(mix)
            results = torch.stack(results, dim=0)  # (num_source, batch, channel, time_step)
            loss = criterion(results, srcs)
            loss = torch.sum(loss, dim=0) / model.num_source
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        loss_history.append(train_loss)
        table.add_row([e, train_loss])

        if e % 5 == 0:
            print(table)
            table.clear_rows()

    print(f'Training [Loss]: {min(loss_history)}')
    plot_metrics(loss_history, 'Loss')

    print('Saving model...')
    save_model(model, 'wave-u-net')
    print('Done!')


@torch.no_grad()
def test(test_loader, model):
    model.eval()

    criterion = MSELoss().to(params.DEVICE)

    train_loss = 0.

    print('=' * 50 + f'Testing in {params.DEVICE}' + '=' * 50)
    for mix, srcs in tqdm(test_loader, desc=f'Testing', colour='green'):
        mix = mix.to(params.DEVICE)
        srcs = torch.stack(srcs, dim=0).to(params.DEVICE)

        results = model(mix)
        results = torch.stack(results, dim=0)  # (num_source, batch, channel, time_step)
        loss = criterion(results, srcs)
        loss = torch.sum(loss, dim=0) / model.num_source
        train_loss += loss.item()

    train_loss /= len(test_loader)
    print(f'Testing [Loss]: {train_loss}')


if __name__ == '__main__':
    set_seed()
    # train_data = MixtureDataset(train=True)
    # train_loader = DataLoader(train_data, batch_size=params.BATCH_SIZE, shuffle=True, num_workers=params.NUM_WORKERS,
    #                           prefetch_factor=params.PREFETCH_FACTOR, pin_memory=True)

    test_data = MixtureDataset(train=False)
    test_loader = DataLoader(test_data, batch_size=params.BATCH_SIZE, shuffle=False)
    model = WaveUNet(in_channels=1, num_source=2, layers=12, Fc=24).to(params.DEVICE)
    # train(train_loader, model)
    load_model(model, 'wave-u-net-20240721-141031.pt')
    test(test_loader, model)
