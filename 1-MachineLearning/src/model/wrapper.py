# -*- coding: UTF-8 -*-

from torchmetrics import MetricCollection

import parameters


class ModelWrapper:

    def __init__(self, model, optimizer, criterion, metrics):
        self.name = model.__class__.__name__
        self.model = model.to(parameters.DEVICE)
        self.criterion = criterion().to(parameters.DEVICE)
        self.metrics = MetricCollection([m.to(parameters.DEVICE) for m in metrics])

        optim_params = parameters.OPTIM_PARAMS[self.name]
        self.optimizer = optimizer(model.parameters(), **optim_params)

    def __iter__(self):
        return iter((self.model, self.optimizer, self.criterion, self.metrics))
