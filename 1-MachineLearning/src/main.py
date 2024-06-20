# -*- coding: UTF-8 -*-
import torch
from torch.nn import MSELoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.regression import MeanAbsoluteError

import parameters
from data import preprocessing, AnimeDataset
from model import AnimeMF
from recorder import Recorder


def train(model, train_loader, validation_loader):
    print('\n' + '=' * 40 + f' Training in {parameters.DEVICE} ' + '=' * 40)

    recorder = Recorder()
    mae_metric = MeanAbsoluteError().to(parameters.DEVICE)

    optimizer = SGD(model.parameters(), lr=parameters.LEARNING_RATE, weight_decay=parameters.WEIGHT_DECAY)
    criterion = MSELoss().to(parameters.DEVICE)

    model.train()
    for e in range(1, parameters.EPOCH + 1):
        train_loss = 0
        validate_loss = 0

        for user_ids, anime_ids, ratings in tqdm(train_loader, desc=f'Training [Epoch {e}/{parameters.EPOCH}]', colour='green'):
            user_ids = user_ids.to(parameters.DEVICE)
            anime_ids = anime_ids.to(parameters.DEVICE)
            ratings = ratings.to(parameters.DEVICE)

            optimizer.zero_grad()
            result = model(user_ids, anime_ids)
            loss = criterion(result, ratings)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            mae_metric.update(result, ratings)

        train_loss /= len(train_loader)
        train_mae = mae_metric.compute().item()
        mae_metric.reset()


        model.eval()
        with torch.no_grad():
            for user_ids, anime_ids, ratings in tqdm(validation_loader, desc=f'Validating [Epoch {e}/{parameters.EPOCH}]', colour='green'):
                user_ids = user_ids.to(parameters.DEVICE)
                anime_ids = anime_ids.to(parameters.DEVICE)
                ratings = ratings.to(parameters.DEVICE)

                result = model(user_ids, anime_ids)
                loss = criterion(result, ratings)

                validate_loss += loss.item()
                mae_metric.update(result, ratings)

            validate_loss /= len(validation_loader)
            validate_mae = mae_metric.compute().item()
            mae_metric.reset()

            recorder.add_record(e, {
                'Loss': {
                    'Train': train_loss,
                    'Validation': validate_loss,
                },
                'MAE': {
                    'Train': train_mae,
                    'Validation': validate_mae
                }
            })

    recorder.show()





if __name__ == '__main__':
    train_data, test_data, validate_data, user_num, anime_num = preprocessing()
    train_dataset = AnimeDataset(train_data)
    test_dataset = AnimeDataset(test_data)
    validate_dataset = AnimeDataset(validate_data)
    train_loader = DataLoader(train_dataset, batch_size=parameters.BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=parameters.BATCH_SIZE, shuffle=False, num_workers=4)
    validate_loader = DataLoader(validate_dataset, batch_size=parameters.BATCH_SIZE, shuffle=False, num_workers=4)

    model = AnimeMF(user_num, anime_num).to(parameters.DEVICE)

    train(model, train_loader, validate_loader)
