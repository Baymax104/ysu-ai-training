# -*- coding: UTF-8 -*-
import torch

import params
from model.decoder import Decoder

if __name__ == '__main__':
    t = torch.ones((params.BATCH_SIZE, 200, 256), dtype=torch.float32)
    model = Decoder(256, 128, 3, params.BATCH_SIZE)
    # model = Encoder(256)
    output = model(t)
    print(output)
