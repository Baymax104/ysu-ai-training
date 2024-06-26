# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from decision_tree import DecisionTree
from graph import export_tree

RANDOM_SEED = 100


def load_data():
    data_df = pd.read_csv('../dataset/iris_data.csv')
    target_df = pd.read_csv('../dataset/iris_target.csv')
    label_df = pd.read_csv('../dataset/iris_label.csv', index_col=0)

    data, target = np.array(data_df), np.array(target_df).squeeze()
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2,
                                                                        random_state=RANDOM_SEED)
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    feature_names = list(data_df.columns)
    class_names = list(label_df['name'])

    return train_data, train_target, test_data, test_target, feature_names, class_names


def accuracy(y_pred, y_true):
    n = y_pred.shape[0]
    return np.sum(y_pred == y_true) / n


if __name__ == '__main__':
    train_data, train_target, test_data, test_target, feature_names, class_names = load_data()
    tree = DecisionTree()
    tree.fit(train_data, train_target)
    pred = tree.predict(test_data)
    acc = accuracy(pred, test_target)

    # Accuracy: 0.8333333333333334
    print(f'Accuracy: {acc}')

    export_tree(tree, feature_names, class_names)
