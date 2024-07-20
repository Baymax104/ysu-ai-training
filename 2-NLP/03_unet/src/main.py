# -*- coding: UTF-8 -*-
from torch.nn import MSELoss
from torch.utils.data import DataLoader

import params
from dataset import MixtureDataset
from model.waveunet import WaveUNet
from utils import set_seed

if __name__ == '__main__':
    set_seed()
    train_data = MixtureDataset(train=True)
    train_loader = DataLoader(train_data, batch_size=params.BATCH_SIZE, shuffle=True, num_workers=params.NUM_WORKERS,
                              prefetch_factor=params.PREFETCH_FACTOR, pin_memory=True)

    model = WaveUNet(in_channels=1, num_source=2, layers=12, Fc=24).to(params.DEVICE)

    mixture, source1, source2 = next(iter(train_loader))

    result = model(mixture)
    source1_result = result[0]
    source2_result = result[1]
    criterion = MSELoss()
    loss = criterion(source1_result, source1)

    print()
