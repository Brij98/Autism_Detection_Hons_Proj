import pandas as pd
import numpy as np

# SVM
#       1/2 * ||W||^2 + C(1/N * sum(max(0, 1 - Yi * (W * Xi +b))))

Learning_rate = 0.00001
Reg_strength = 1000  # C


def read_csv_data(fl_dir):
    dataframe = pd.read_csv(fl_dir)

    # replacing the label ASD with -1.
    # replacing the label TD with 1.
    dataframe["feature_class"]. \
        replace({"ASD": -1, "TD": 1},
                inplace=True)

    # shuffling data
    # splitting dataframe into train and test
    train_set = dataframe.sample(frac=0.70, random_state=1)
    test_set = dataframe.drop(train_set.index)
    test_set = test_set.sample(frac=1)

    dataframe = None

    # extract the label column from train set
    y_train = train_set.iloc[:, -1]

    # extract the features from train set
    x_train = train_set.drop(labels="feature_class", axis=1)
    #  adding a new col to the train feature
    x_train.insert(loc=len(x_train.columns), column='intercept', value=1)

    # extract the label column from test set
    y_test = test_set.iloc[:, -1]

    # extract the features from test set
    x_test = test_set.drop(labels="feature_class", axis=1)
    #  adding a new col to test feature
    x_test.insert(loc=len(x_test.columns), column='intercept', value=1)

    # print(x_test)  # debug
    # print(y_test)  # debug

    # apply normalization to the features

    return x_train, y_train, x_test, y_test


#  1/2 * ||W||^2 + C(1/N * sum(max(0, 1 - Yi * (W * Xi +b))))
def calculate_hinge_loss(vector_w, x, y):
    num_rows = x.shape[0]
    distances = []
    for xi, yi in zip(x, y):
        #  calculating: 1 - Yi * (w * Xi)
        dist = 1 - yi * (np.dot(xi, vector_w))
        #  max(0, dist)
        if dist < 0:
            distances.append(0)
        else:
            distances.append(dist)

    # calculate hinge loss
    # C(1/N * sum(max(0, 1 - Yi * (W * Xi +b))))
    hinge_loss = Reg_strength * (np.sum(distances) / num_rows)

    # 1/2 * ||W||^2 + hinge_loss
    ret_val = 0.5 * np.dot(vector_w, vector_w) + hinge_loss
    return ret_val


class SupportVectorMachine:
    def __init__(self):
        pass


if __name__ == "__main__":
    svm = SupportVectorMachine()
    read_csv_data("D:/TrainingDataset_YEAR_PROJECT/TrainingSet.csv")
