# -*- coding: UTF-8 -*-

import numpy as np


class LinearRegression:

    def __init__(self, feature_dim, sample_num, learning_rate):
        self.sample_num = sample_num
        self.features_dim = feature_dim
        self.lr = learning_rate
        self.weights = np.ones((feature_dim, 1))
        self.bias = 0

    def fit(self, x, target):
        # predicate
        y_hat = x @ self.weights + self.bias

        # gradient
        dw = (x.T @ (y_hat - target)) / self.sample_num
        db = np.sum(y_hat - target) / self.sample_num

        # update parameter
        self.weights -= self.lr * dw
        self.bias -= self.lr * db

        return y_hat

    def predict(self, x):
        return x @ self.weights + self.bias
