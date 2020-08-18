import random

import pandas as pd


def train_test_split(dataframe, testsize):
    if isinstance(testsize, float):
        testsize = round(testsize * len(dataframe))

    indices = dataframe.index.tolist()
    test_indcs = random.sample(population=indices, k=testsize)

    x_test = dataframe.loc[test_indcs]
    x_train = dataframe.drop(test_indcs)

    y_train = x_train.iloc[:, -1]
    x_train = x_train.drop(labels='feature_class', axis=1)

    y_test = x_test.iloc[:, -1]
    x_test = x_test.drop(labels='feature_class', axis=1)

    return x_train, x_test, y_train, y_test


def calc_accuracy_score(y_true, y_pred):
    tp_tn = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp_tn += 1

    return tp_tn / y_true.shape[0]


def calc_confusion_matrix(y_true, y_pred):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for true, pred in zip(y_true, y_pred):
        if true > 0:
            if true == pred:
                true_pos += 1
            else:
                true_neg += 1
        else:
            if true < 0:
                if true == pred:
                    false_neg += 1
                else:
                    false_pos += 1

    return true_pos, true_neg, false_pos, false_neg
