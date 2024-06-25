# -*- coding: UTF-8 -*-

import numpy as np


def distance(x, y):
    return np.linalg.norm(x - y)


class KMeans:

    def __init__(self, k, data):
        centers_idx = np.random.choice(data.shape[0], k, replace=False)
        self.centers = data[centers_idx]
        self.k = k

    def fit(self, X):
        clusters = [[] for _ in range(self.k)]
        closest_class = []
        for x in X:
            distances = [distance(x, c) for c in self.centers]
            closest_idx = np.argmin(distances, axis=0)
            clusters[closest_idx].append(x)
            closest_class.append(closest_idx)

        self.centers = np.array([np.mean(cluster, axis=0) for cluster in clusters])
        closest_class = np.array(closest_class)
        return closest_class

    def predict(self, X):
        closest_class = []
        for x in X:
            distances = [distance(x, c) for c in self.centers]
            closest_class.append(np.argmin(distances, axis=0))

        return np.array(closest_class)
