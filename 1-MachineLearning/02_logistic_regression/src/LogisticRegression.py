# -*- coding: UTF-8 -*-
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:

    def __init__(self, feature_dim, sample_num, learning_rate):
        self.feature_dim = feature_dim
        self.sample_num = sample_num
        self.lr = learning_rate
        self.weights = np.ones((feature_dim + 1, 1))

    def fit(self, X, y):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        y_hat = sigmoid(X @ self.weights)

        # gradient descent
        grad = (1 / self.sample_num) * (X.T @ (y_hat - y))
        self.weights -= self.lr * grad
        return y_hat


    def predict(self, X):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        return sigmoid(X @ self.weights)
