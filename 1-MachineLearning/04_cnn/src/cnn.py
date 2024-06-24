# -*- coding: UTF-8 -*-

import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  # (1, 28, 28) -> (32, 28, 28)
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (32, 28, 28) -> (32, 14, 14)
            nn.Flatten(),  # (32, 14, 14) -> (32 * 14 * 14,)
            nn.Linear(in_features=32 * 14 * 14, out_features=1024),  # (6272,) -> (1024,)
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=10),  # (1024,) -> (10,)
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


