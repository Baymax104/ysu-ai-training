# -*- coding: UTF-8 -*-

from typing import Annotated

import typer
from accelerate import Accelerator
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MetricCollection, F1Score
from tqdm import tqdm
from transformers import AutoTokenizer

import params
from dataset import WMTDataset, Collator
from model.transformer import Transformer
from recorder import Recorder
from utils import *

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def train():
    tokenizer = AutoTokenizer.from_pretrained(params.TOKENIZER_NAME)
    dataset = WMTDataset(split='train', data_dir=params.DATA_DIR)
    dataloader = DataLoader(
        dataset,
        batch_size=params.BATCH_SIZE,
        shuffle=True,
        num_workers=params.NUM_WORKERS,
        prefetch_factor=params.PREFETCH_FACTOR,
        collate_fn=Collator(tokenizer, params.MAX_LENGTH)
    )
    model = Transformer(
        num_embeddings=tokenizer.vocab_size,
        max_length=params.MAX_LENGTH,
        **params.TRANSFORMER_PARAMETERS
    ).to(params.DEVICE)

    criterion = CrossEntropyLoss().to(params.DEVICE)
    optimizer = Adam(model.parameters(), lr=params.LR, weight_decay=params.WEIGHT_DECAY)
    accelerator = Accelerator(mixed_precision='fp16', cpu=params.IS_CPU)

    recorder = Recorder(['Loss', 'Accuracy', 'F1-Score'], steps=params.EPOCHS, log_dir=params.LOG_DIR)
    metrics = MetricCollection({
        'Accuracy': Accuracy('multiclass', num_classes=tokenizer.vocab_size),
        'F1-Score': F1Score('multiclass', num_classes=tokenizer.vocab_size)
    }).to(params.DEVICE)

    model, optimizer, criterion, dataloader = accelerator.prepare([model, optimizer, criterion, dataloader])

    print('=' * 50 + f'Training in {params.DEVICE}' + '=' * 50)
    model.train()
    for e in range(1, params.EPOCHS + 1):
        train_loss = 0.

        for batch in tqdm(dataloader, desc=f'Training [{e}/{params.EPOCHS}]', colour='green'):
            input_ids = batch['input_ids'].to(params.DEVICE)
            input_attention_mask = batch['input_attention_mask'].to(params.DEVICE)
            output_ids = batch['output_ids'].to(params.DEVICE)
            output_attention_mask = batch['output_attention_mask'].to(params.DEVICE)
            causal_mask = batch['causal_mask'].to(params.DEVICE)

            with accelerator.autocast():
                result = model(input_ids, output_ids, input_attention_mask, output_attention_mask, causal_mask)
                result = result.transpose(-1, -2)
                loss = criterion(result, output_ids)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            metrics.update(result, output_ids)

        train_loss /= len(dataloader)
        train_metrics = metrics.compute()
        train_metrics['Loss'] = train_loss

        recorder.add_record(train_metrics, step=e)

        if e % 5 == 0:
            recorder.print(clean=True)

    recorder.plot(model_title='Transformer')

    print('Saving model...')
    save_model(model, 'transformer', params.LOG_DIR)
    print('Done!')


@torch.no_grad()
@app.command()
def test(model_file: Annotated[Path, typer.Argument(exists=True, file_okay=True)]):
    tokenizer = AutoTokenizer.from_pretrained(params.TOKENIZER_NAME)
    dataset = WMTDataset(split='test', data_dir=params.DATA_DIR)
    dataloader = DataLoader(
        dataset,
        batch_size=params.BATCH_SIZE,
        shuffle=True,
        num_workers=params.NUM_WORKERS,
        prefetch_factor=params.PREFETCH_FACTOR,
        collate_fn=Collator(tokenizer, params.MAX_LENGTH)
    )
    model = Transformer(
        num_embeddings=tokenizer.vocab_size,
        max_length=params.MAX_LENGTH,
        **params.TRANSFORMER_PARAMETERS
    ).to(params.DEVICE)
    criterion = CrossEntropyLoss().to(params.DEVICE)
    load_model(model, model_file)

    recorder = Recorder(['Loss', 'Accuracy', 'F1-Score'], steps=params.EPOCHS, log_dir=params.LOG_DIR)
    metrics = MetricCollection({
        'Accuracy': Accuracy('multiclass', num_classes=tokenizer.vocab_size),
        'F1-Score': F1Score('multiclass', num_classes=tokenizer.vocab_size)
    }).to(params.DEVICE)

    print('=' * 50 + f'Testing in {params.DEVICE}' + '=' * 50)
    model.eval()
    test_loss = 0.
    for batch in tqdm(dataloader, desc=f'Testing', colour='green'):
        input_ids = batch['input_ids'].to(params.DEVICE)
        input_attention_mask = batch['input_attention_mask'].to(params.DEVICE)
        output_ids = batch['output_ids'].to(params.DEVICE)
        output_attention_mask = batch['output_attention_mask'].to(params.DEVICE)
        causal_mask = batch['causal_mask'].to(params.DEVICE)

        result = model(input_ids, output_ids, input_attention_mask, output_attention_mask, causal_mask)
        result = result.transpose(-1, -2)
        loss = criterion(result, output_ids)

        test_loss += loss.item()
        metrics.update(result, output_ids)

    test_loss /= len(dataloader)
    test_metrics = metrics.compute()
    test_metrics['Loss'] = test_loss
    recorder.add_record(test_metrics)
    recorder.print()


if __name__ == '__main__':
    set_seed(params.RANDOM_SEED)
    app()
