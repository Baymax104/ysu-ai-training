# -*- coding: UTF-8 -*-
from argparse import ArgumentParser

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Accuracy, F1Score, Precision, Recall
from tqdm import tqdm

import params
import utils
from dataset import SST2Dataset, Collator
from model import DistilBertModel
from recorder import Recorder


def train():
    dataset = SST2Dataset(train=True)
    loader = DataLoader(dataset, batch_size=params.BATCH_SIZE, shuffle=True, collate_fn=Collator(dataset.tokenizer),
                        num_workers=params.NUM_WORKERS, prefetch_factor=params.PREFETCH_FACTOR, pin_memory=True)

    recorder = Recorder(['Loss', 'Accuracy', 'F1-Score', 'Precision', 'Recall'], steps=params.EPOCHS)
    metrics = MetricCollection({
        'Accuracy': Accuracy('binary'),
        'F1-Score': F1Score('binary'),
        'Precision': Precision('binary'),
        'Recall': Recall('binary'),
    }).to(params.DEVICE)

    model = DistilBertModel().to(params.DEVICE)
    criterion = CrossEntropyLoss().to(params.DEVICE)
    optimizer = AdamW(model.parameters(), lr=params.LR, weight_decay=params.WEIGHT_DECAY)

    print('=' * 50 + f'Training in {params.DEVICE}' + '=' * 50)
    for e in range(1, params.EPOCHS + 1):
        train_loss = 0.
        for batch in tqdm(loader, desc=f'Training Epoch[{e}/{params.EPOCHS}]', colour='green'):
            input_ids = batch['input_ids'].to(params.DEVICE)
            attention_mask = batch['attention_mask'].to(params.DEVICE)
            labels = batch['labels'].to(params.DEVICE)

            result = model(input_ids, attention_mask)
            loss = criterion(result, labels)

            train_loss += loss.item()
            metrics.update(result, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(loader)
        train_metrics = metrics.compute()
        train_metrics['Loss'] = train_loss
        recorder.add_record(train_metrics, step=e)

        if e % 5 == 0:
            recorder.print(clean=True)

    for metric, best_value in recorder.get_best(['min', 'max', 'max', 'max', 'max']).items():
        print(f'Training [{metric}]: {best_value}')
    recorder.plot(params.LOG_DIR, model_title='DistilBert')

    print('Saving model...')
    utils.save_model(model, 'distilbert')
    print('Done!')


@torch.no_grad()
def test(model_file):
    dataset = SST2Dataset(train=False)
    loader = DataLoader(dataset, batch_size=params.BATCH_SIZE, shuffle=False,
                        collate_fn=Collator(dataset.tokenizer), num_workers=params.NUM_WORKERS)

    recorder = Recorder(['Loss', 'Accuracy', 'F1-Score', 'Precision', 'Recall'])
    metrics = MetricCollection({
        'Accuracy': Accuracy('binary'),
        'F1-Score': F1Score('binary'),
        'Precision': Precision('binary'),
        'Recall': Recall('binary'),
    }).to(params.DEVICE)

    model = DistilBertModel().to(params.DEVICE)
    utils.load_model(model, model_file)
    criterion = CrossEntropyLoss().to(params.DEVICE)

    print('=' * 50 + f'Testing in {params.DEVICE}' + '=' * 50)
    test_loss = 0.
    for batch in tqdm(loader, desc=f'Testing', colour='green'):
        input_ids = batch['input_ids'].to(params.DEVICE)
        attention_mask = batch['attention_mask'].to(params.DEVICE)
        labels = batch['labels'].to(params.DEVICE)

        result = model(input_ids, attention_mask)
        loss = criterion(result, labels)

        test_loss += loss.item()
        metrics.update(result, labels)

    test_loss /= len(loader)
    test_metrics = metrics.compute()
    test_metrics['Loss'] = test_loss
    recorder.add_record(test_metrics)
    recorder.print()


if __name__ == '__main__':
    utils.set_seed(params.RANDOM_SEED)
    parser = ArgumentParser()
    parser.add_argument('--test', type=str)
    args = parser.parse_args()

    if args.test:
        test(args.test)
    else:
        train()
