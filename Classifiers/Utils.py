import random
import numpy as np
import pandas as pd


def train_test_split(dataframe, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(dataframe))

    # storing the indices of the data frame into a list
    indices = dataframe.index.tolist()
    test_indcs = random.sample(population=indices, k=test_size)

    #  extract the randomly generated indices
    x_test = dataframe.loc[test_indcs]
    x_train = dataframe.drop(test_indcs)

    y_train = x_train.iloc[:, -1]
    x_train = x_train.drop(x_train.columns[-1], axis=1)

    y_test = x_test.iloc[:, -1]
    x_test = x_test.drop(x_test.columns[-1], axis=1)

    return x_train, x_test, y_train, y_test


def generate_classification_report(y_true, y_pred):
    pass


def calculate_accuracy_score(y_true, y_pred):
    # true positive and true negative
    tp_tn = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp_tn += 1

    return tp_tn / y_true.shape[0]


def calculate_confusion_matrix(y_actual, y_pred):
    # Pos = ASD, Neg = Normal
    # actual val Pos && pred val Pos
    true_pos = 0
    # actual val Neg && pred val Neg
    true_neg = 0
    # actual val Neg && pred val Pos
    false_pos = 0
    # actual val Neg && pred Neg
    false_neg = 0

    # val > 0 : ASD, val <= 0 Normal
    for actual_val, pred_val in zip(y_actual, y_pred):
        if actual_val > 0:
            if actual_val == pred_val:
                true_pos += 1
            else:
                false_neg += 1
        else:
            if pred_val > 0:
                false_pos += 1
            else:
                true_neg += 1

    return [[true_pos, false_neg], [false_pos, true_neg]]


# finds the min and max val for each col
def calculate_min_max_scalar(dataset):
    min_max_list = list()

    min_max_list.append(dataset.min())
    min_max_list.append(dataset.max())

    return min_max_list


# for each column represent feature values in the range of 0 - 1
def normalize_dataset(df, min_max):
    col_names = list(df.columns)

    for col_name in col_names:
        min_val = min_max[0][col_name]
        max_val = min_max[1][col_name]

        df[col_name] = (df[col_name] - min_val) / (max_val - min_val)

    return df


def plot_roc_curve(y_pred, y_true):
    pass
