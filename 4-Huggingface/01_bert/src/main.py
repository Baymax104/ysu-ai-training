# -*- coding: UTF-8 -*-
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from tqdm import tqdm

import params
import utils
from dataset import ImdbDataset
from model import BertClassifier
from recorder import Recorder


def train(train_loader, model):
    model.train()

    criterion = CrossEntropyLoss().to(params.DEVICE)
    optimizer = Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=params.LR,
                     weight_decay=params.WEIGHT_DECAY)

    recorder = Recorder(['Loss', 'Accuracy', 'F1-Score'], steps=params.EPOCHS)
    metrics = MetricCollection({
        'Accuracy': BinaryAccuracy(),
        'F1-Score': BinaryF1Score(),
    }).to(params.DEVICE)

    print('=' * 50 + f'Training in {params.DEVICE}' + '=' * 50)
    for e in range(1, params.EPOCHS + 1):
        train_loss = 0.
        for batch in tqdm(train_loader, desc=f'Training [{e}/{params.EPOCHS}]', colour='green'):
            input_ids = batch['input_ids'].to(params.DEVICE)
            attention_mask = batch['attention_mask'].to(params.DEVICE)
            token_type_ids = batch['token_type_ids'].to(params.DEVICE)
            label = batch['label'].to(params.DEVICE)

            result = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(result, label)

            train_loss += loss.item()
            sigmoid_result = torch.sigmoid(result.detach())
            metrics.update(sigmoid_result, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_metrics = metrics.compute()
        train_metrics['Loss'] = train_loss
        recorder.add_record(train_metrics, e)

        if e % 5 == 0:
            recorder.print(clean=True)

    print(f'Training [Loss]: {min(recorder["Loss"])}')
    print(f'Training [Accuracy]: {max(recorder["Accuracy"])}')
    print(f'Training [F1-Score]: {max(recorder["F1-Score"])}')
    recorder.plot(model_title='BertClassifier')

    print('Saving model...')
    utils.save_model(model, 'bert-classifier')
    print('Done!')


@torch.no_grad()
def test(test_loader, model):
    model.eval()

    criterion = CrossEntropyLoss().to(params.DEVICE)

    recorder = Recorder(['Loss', 'Accuracy', 'F1-Score'])
    metrics = MetricCollection({
        'Accuracy': BinaryAccuracy(),
        'F1-Score': BinaryF1Score(),
    }).to(params.DEVICE)

    print('=' * 50 + f'Testing in {params.DEVICE}' + '=' * 50)
    test_loss = 0.
    for batch in tqdm(test_loader, desc=f'Testing', colour='green'):
        input_ids = batch['input_ids'].to(params.DEVICE)
        attention_mask = batch['attention_mask'].to(params.DEVICE)
        token_type_ids = batch['token_type_ids'].to(params.DEVICE)
        label = batch['label'].to(params.DEVICE)

        result = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(result, label)

        test_loss += loss.item()
        sigmoid_result = torch.sigmoid(result.detach())
        metrics.update(sigmoid_result, label)

    test_loss /= len(test_loader)
    test_metrics = metrics.compute()
    test_metrics['Loss'] = test_loss
    recorder.add_record(test_metrics)
    recorder.print()


if __name__ == '__main__':
    utils.set_seed(params.RANDOM_SEED)
    # train_data = ImdbDataset(train=True)
    # train_loader = DataLoader(train_data, batch_size=params.BATCH_SIZE, shuffle=True, num_workers=params.NUM_WORKERS,
    #                           prefetch_factor=params.PREFETCH_FACTOR, pin_memory=True)

    test_data = ImdbDataset(train=False)
    test_loader = DataLoader(test_data, batch_size=params.BATCH_SIZE, shuffle=False, num_workers=params.NUM_WORKERS)

    model = BertClassifier(num_classes=2).to(params.DEVICE)
    utils.load_model(model, 'bert-classifier-20240802-020404.pt')

    # train(train_loader, model)
    test(test_loader, model)
