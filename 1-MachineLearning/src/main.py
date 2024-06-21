# -*- coding: UTF-8 -*-
import torch
from torch.nn import MSELoss
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from tqdm import tqdm

import parameters
from data import preprocessing, AnimeDataset
from model.gmf import GMF
from model.mf import MF
from model.mlp import MLP
from model.wrapper import ModelWrapper
from recorder import Recorder


def train(train_loader, validation_loader, model_wrapper):
    print('\n' + '=' * 40 + f' Training in {parameters.DEVICE} ' + '=' * 40)

    recorder = Recorder(model_wrapper.name)

    model, optimizer, criterion, metrics = model_wrapper

    model.train()
    for e in range(1, parameters.EPOCH + 1):
        train_loss = 0
        validate_loss = 0

        # batch train
        for user_ids, anime_ids, ratings in tqdm(train_loader, desc=f'Training [Epoch {e}/{parameters.EPOCH}]',
                                                 colour='green'):
            user_ids = user_ids.to(parameters.DEVICE)
            anime_ids = anime_ids.to(parameters.DEVICE)
            ratings = ratings.to(parameters.DEVICE)

            optimizer.zero_grad()
            result = model(user_ids, anime_ids)
            loss = criterion(result, ratings)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            metrics.update(result, ratings)

        # record train metrics
        train_loss /= len(train_loader)
        train_metrics = metrics.compute()
        train_metrics['Loss'] = train_loss
        metrics.reset()
        recorder.add_train_record(e, train_metrics)


        model.eval()
        with torch.no_grad():
            # batch validate
            for user_ids, anime_ids, ratings in tqdm(validation_loader,
                                                     desc=f'Validating [Epoch {e}/{parameters.EPOCH}]', colour='green'):
                user_ids = user_ids.to(parameters.DEVICE)
                anime_ids = anime_ids.to(parameters.DEVICE)
                ratings = ratings.to(parameters.DEVICE)

                result = model(user_ids, anime_ids)
                loss = criterion(result, ratings)

                validate_loss += loss.item()
                metrics.update(result, ratings)

            # record validate metrics
            validate_loss /= len(validation_loader)
            validate_metrics = metrics.compute()
            validate_metrics['Loss'] = validate_loss
            metrics.reset()
            recorder.add_validate_record(e, validate_metrics)

    recorder.show()


if __name__ == '__main__':
    train_data, test_data, validate_data, user_num, anime_num = preprocessing()
    train_dataset = AnimeDataset(train_data)
    test_dataset = AnimeDataset(test_data)
    validate_dataset = AnimeDataset(validate_data)
    train_loader = DataLoader(train_dataset, batch_size=parameters.BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=parameters.BATCH_SIZE, shuffle=False, num_workers=4)
    validate_loader = DataLoader(validate_dataset, batch_size=parameters.BATCH_SIZE, shuffle=False, num_workers=4)

    mf_wrapper = ModelWrapper(
        model=MF(user_num, anime_num),
        optimizer=SGD,
        criterion=MSELoss,
        metrics=[MeanAbsoluteError(), MeanSquaredError(squared=False)]
    )

    gmf_wrapper = ModelWrapper(
        model=GMF(user_num, anime_num),
        optimizer=SGD,
        criterion=MSELoss,
        metrics=[MeanAbsoluteError(), MeanSquaredError(squared=False)]
    )

    mlp_wrapper = ModelWrapper(
        model=MLP(user_num, anime_num),
        optimizer=SGD,
        criterion=MSELoss,
        metrics=[MeanAbsoluteError(), MeanSquaredError(squared=False)]
    )

    train(train_loader, validate_loader, gmf_wrapper)
