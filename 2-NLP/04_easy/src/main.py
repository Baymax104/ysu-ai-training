# -*- coding: UTF-8 -*-
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score, MultilabelRecall, MultilabelPrecision
from tqdm import tqdm

import params
from dataset import EasyDataset
from model import BertMultiClassification
from recorder import Recorder
from utils import set_seed, save_model, load_model


def train(train_loader, model):
    model.train()

    criterion = BCEWithLogitsLoss().to(params.DEVICE)
    optimizer = Adam(model.parameters(), lr=params.LR, weight_decay=params.WEIGHT_DECAY)

    recorder = Recorder(['Loss', 'Accuracy', 'Recall', 'Precision', 'F1-Score'], steps=params.EPOCH)
    metrics = MetricCollection({
        'Accuracy': MultilabelAccuracy(num_labels=model.num_classes),
        'Recall': MultilabelRecall(num_labels=model.num_classes),
        'Precision': MultilabelPrecision(num_labels=model.num_classes),
        'F1-Score': MultilabelF1Score(num_labels=model.num_classes),
    }).to(params.DEVICE)

    print('=' * 50 + f'Training in {params.DEVICE}' + '=' * 50)
    for e in range(1, params.EPOCH + 1):
        train_loss = 0.
        for b, sample in enumerate(tqdm(train_loader, desc=f'Training Epoch [{e}/{params.EPOCH}]', colour='green')):
            input_ids = sample['input_ids'].to(params.DEVICE)
            attention_mask = sample['attention_mask'].to(params.DEVICE)
            token_type_ids = sample['token_type_ids'].to(params.DEVICE)
            label = sample['label'].to(params.DEVICE)

            optimizer.zero_grad()
            result = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(result, label)

            loss.backward()
            optimizer.step()

            result_sigmoid = torch.sigmoid(result).detach()
            train_loss += loss.item()
            metrics.update(result_sigmoid, label)

        train_loss /= len(train_loader)
        epoch_metrics = metrics.compute()
        epoch_metrics['Loss'] = train_loss
        recorder.add_record(epoch_metrics, step=e)

        if e % 5 == 0:
            recorder.print(clean=True)

    print(f'Training [Loss]: {min(recorder["Loss"])}')
    print(f'Training [Accuracy]: {max(recorder["Accuracy"])}')
    print(f'Training [F1-Score]: {max(recorder["F1-Score"])}')
    print(f'Training [Recall]: {max(recorder["Recall"])}')
    print(f'Training [Precision]: {max(recorder["Precision"])}')
    recorder.plot(model_title='BertMultiClassification')

    print('Saving model...')
    save_model(model, 'bert-multiclassification')
    print('Done!')


@torch.no_grad()
def test(test_loader, model):
    model.eval()
    criterion = BCEWithLogitsLoss().to(params.DEVICE)

    recorder = Recorder(['Loss', 'Accuracy', 'Recall', 'Precision', 'F1-Score'])
    metrics = MetricCollection({
        'Accuracy': MultilabelAccuracy(num_labels=model.num_classes),
        'Recall': MultilabelRecall(num_labels=model.num_classes),
        'Precision': MultilabelPrecision(num_labels=model.num_classes),
        'F1-Score': MultilabelF1Score(num_labels=model.num_classes)
    }).to(params.DEVICE)

    print('=' * 50 + f'Testing in {params.DEVICE}' + '=' * 50)
    test_loss = 0.
    for sample in tqdm(test_loader, desc=f'Testing', colour='green'):
        input_ids = sample['input_ids'].to(params.DEVICE)
        attention_mask = sample['attention_mask'].to(params.DEVICE)
        token_type_ids = sample['token_type_ids'].to(params.DEVICE)
        label = sample['label'].to(params.DEVICE)

        result = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(result, label)

        result_sigmoid = torch.sigmoid(result).detach()
        test_loss += loss.item()
        metrics.update(result_sigmoid, label)

    test_loss /= len(test_loader)
    test_metrics = metrics.compute()
    test_metrics['Loss'] = test_loss
    recorder.add_record(test_metrics)
    recorder.print()


if __name__ == '__main__':
    set_seed(params.RANDOM_SEED)
    # train_data = EasyDataset(train=True)
    test_data = EasyDataset(train=False)
    # train_loader = DataLoader(train_data, batch_size=params.BATCH_SIZE, shuffle=True, num_workers=params.NUM_WORKERS,
    #                           prefetch_factor=params.PREFETCH_FACTOR, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=params.BATCH_SIZE, shuffle=False, num_workers=params.NUM_WORKERS)

    model = BertMultiClassification(num_classes=40).to(params.DEVICE)
    # train(train_loader, model)
    load_model(model, 'bert-multiclassification-20240721-225634.pt')
    test(test_loader, model)
