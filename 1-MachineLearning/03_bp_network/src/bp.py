# -*- coding: UTF-8 -*-

import torch


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def MSE(y_pred, y_true):
    n = y_pred.shape[0]
    return torch.sum((y_pred - y_true) ** 2) / (2 * n)


class BP:

    def __init__(self, lr):
        self.lr = lr
        self.layers = [
            Linear(784, 128, sigmoid, sigmoid_prime),
            Linear(128, 10, sigmoid, sigmoid_prime)
        ]

    def fit(self, x, y):
        n = x.shape[0]
        y_hat = self.__forward(x)
        error = (y_hat - y) / n
        self.__backward(error)
        return y_hat

    def predict(self, x):
        return self.__forward(x)

    def __forward(self, x):
        for layer in self.layers:
            x = layer.compute(x)
        return x

    def __backward(self, error):
        for layer in reversed(self.layers):
            error = layer.update(error, self.lr)


class Linear:

    def __init__(self, input_size, output_size, activation_func, activation_derivative_func):
        self.activation_func = activation_func
        self.activation_derivative = activation_derivative_func
        self.x = None
        self.y = None
        self.y_activate = None

        self.weights = torch.randn(input_size, output_size) / torch.sqrt(torch.tensor(input_size))
        self.bias = torch.randn(output_size) / torch.sqrt(torch.tensor(input_size))

    def compute(self, x):
        self.x = x
        self.y = torch.matmul(x, self.weights) + self.bias
        self.y_activate = self.activation_func(self.y)
        return self.y_activate

    def update(self, error, lr):
        if self.x is None or self.y is None or self.y_activate is None:
            raise ValueError('parameters are null, call fit() first')

        dy = error * self.activation_derivative(self.y)  # 误差对当前层线性输出的偏导
        dw = torch.matmul(self.x.T, dy)  # 误差对参数的偏导，等于误差对线性输出的偏导*线性输出对参数的偏导
        db = torch.sum(dy, dim=0)
        # 更新参数
        self.weights -= lr * dw
        self.bias -= lr * db
        error = torch.matmul(dy, self.weights.T)  # 前一层误差
        return error
