# -*- coding: UTF-8 -*-

import numpy as np


def info_entropy(target):
    entropy = 0
    for c in set(target):
        p = len(target[target == c]) / len(target)
        entropy -= p * np.log2(p)
    return entropy


def info_gain(target, left_target, right_target):
    left_ratio = len(left_target) / len(target)
    left_entropy = info_entropy(left_target)
    right_ratio = len(right_target) / len(target)
    right_entropy = info_entropy(right_target)
    entropy = info_entropy(target)
    return entropy - (left_ratio * left_entropy + right_ratio * right_entropy)


class Node:
    def __init__(self, value=None, feature=None, threshold=None, left=None, right=None):
        self.value = value  # target class
        self.feature = feature  # 0, 1, 2, 3
        self.threshold = threshold  # split threshold
        self.left = left  # [(data, target)]
        self.right = right  # [(data, target)]


class DecisionTree:

    def __init__(self):
        self.root = None

    @staticmethod
    def __split(data_target, feature):
        data = np.array([x for x, _ in data_target])
        threshold = np.median(data[:, feature])
        left = [(x, y) for x, y in data_target if x[feature] <= threshold]
        right = [(x, y) for x, y in data_target if x[feature] > threshold]
        return left, right, threshold

    @staticmethod
    def __chose_best_feature(data_target, feature_list):
        best_feature = -1
        best_gain = -1
        target = np.array([y for _, y in data_target])
        for feature in feature_list:
            left, right, _ = DecisionTree.__split(data_target, feature)
            left_target = np.array([y for _, y in left])
            right_target = np.array([y for _, y in right])
            gain = info_gain(target, left_target, right_target)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature

        return best_feature

    @staticmethod
    def __build_tree(data_target, feature_list):
        target = [y for _, y in data_target]

        # 若当前节点类别全相同，返回叶子节点
        if target[1:] == target[:-1]:
            return Node(value=target[0])

        # 若属性集为空，选择个数最多的类别
        if len(feature_list) == 0:
            target = np.array(target)
            counts = np.bincount(target)
            most_class = np.argmax(counts)
            return Node(value=most_class)

        # 构造子树
        best_feature = DecisionTree.__chose_best_feature(data_target, feature_list)
        left, right, threshold = DecisionTree.__split(data_target, best_feature)
        feature_list.remove(best_feature)

        left_root = DecisionTree.__build_tree(left, feature_list)
        right_root = DecisionTree.__build_tree(right, feature_list)

        return Node(feature=best_feature, threshold=threshold, left=left_root, right=right_root)

    @staticmethod
    def __trace(node, x):
        # 到达叶子节点
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return DecisionTree.__trace(node.left, x)
        else:
            return DecisionTree.__trace(node.right, x)

    def fit(self, x, y):
        data_target = list(zip(x, y))
        feature_list = list(range(x.shape[1]))
        self.root = self.__build_tree(data_target, feature_list)

    def predict(self, X):
        pred = []
        # serial predict
        for x in X:
            pred.append(self.__trace(self.root, x))
        return np.array(pred)
