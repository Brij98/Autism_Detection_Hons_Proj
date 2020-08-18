import numpy as np
import pandas as pd
# SVM
#       1/2 * ||W||^2 + C(1/N * sum(max(0, 1 - Yi * (W * Xi +b))))
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

Learning_rate = 0.000001
Reg_strength = 10000  # C


def proc_CSV_data(fl_dir):
    dataframe = pd.read_csv(fl_dir)

    # replacing the label ASD with -1.
    # replacing the label TD with 1.
    dataframe["feature_class"]. \
        replace({"ASD": -1.0, "TD": 1.0},
                inplace=True)

    # print(dataframe)

    # shuffling data
    dataframe = dataframe.sample(frac=1)
    # splitting dataframe into train and test
    train_set = dataframe.sample(frac=0.80, random_state=1)
    test_set = dataframe.drop(train_set.index)
    test_set = test_set.sample(frac=1)

    dataframe = None

    # extract the label column from train set
    y_train = train_set.iloc[:, -1]

    # extract the features from train set
    x_train = train_set.drop(labels="feature_class", axis=1)
    #  adding a new col to the train feature
    # x_train.insert(loc=len(x_train.columns), column='intercept', value=1)

    # extract the label column from test set
    y_test = test_set.iloc[:, -1]

    # extract the features from test set
    x_test = test_set.drop(labels="feature_class", axis=1)
    #  adding a new col to test feature
    # x_test.insert(loc=len(x_test.columns), column='intercept', value=1)

    # print(type(x_train))  # debug
    # print(type(y_train))  # debug

    # normalizing the  Data
    scalar = MinMaxScaler()
    x_train = pd.DataFrame(scalar.fit_transform(x_train.values))
    x_test = pd.DataFrame(scalar.transform(x_test.values))

    # x_train.insert(loc=len(x_train.columns), column='intercept', value=1)
    # x_test.insert(loc=len(x_test.columns), column='intercept', value=1)

    return x_train, x_test, y_train, y_test


#  1/2 * ||W||^2 + C(1/N * sum(max(0, 1 - Yi * (W * Xi +b))))
def calc_hinge_loss(vector_w, x, y):
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


def calc_cost_gradient(w, x, y):
    # x = np.array([x])
    # y = np.array([y])

    dist = 1 - (y * np.dot(w, x))
    dist_w = np.zeros(len(w))

    if max(0, dist) == 0:
        dist_w += w
    else:
        dist_w += w - (Reg_strength * y * x)

    return dist_w


def stochastic_gradient_descent(features, labels):
    global Learning_rate
    max_epochs = 17000
    weights = np.zeros(features.shape[1])
    prev_cost = float("inf")
    cost_threshold = 0.01
    count = 0

    for epoch in range(1, max_epochs):
        X, Y = shuffle_data(features=features, labels=labels)
        for indx, x in enumerate(X):
            ascent = calc_cost_gradient(w=weights, x=x, y=Y[indx])
            weights = weights - (Learning_rate * ascent)

        if epoch == 2 ** count or epoch == max_epochs - 1:
            cost = calc_hinge_loss(vector_w=weights, x=features, y=labels)
            print("Epoch No.: {} || Cost: {}".format(epoch, cost))

            # stopping condition
            # if abs(prev_cost - cost) < cost_threshold * prev_cost:
            #     return weights
            # prev_cost = cost
            count += 1
            #  Learning_rate /= 10
    return weights


#  shuffle features and labels in order
def shuffle_data(features, labels):
    index = np.random.permutation(len(labels))
    x = []
    y = []

    for i in index:
        x.append(features[i])
        y.append(labels[i])
    return x, y


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


def test_SVM(weights, x_test, y_test):
    y_test_predicted = np.array([])

    for i in range(x_test.shape[0]):
        y_predict = np.sign(np.dot(x_test[i], weights))
        y_test_predicted = np.append(y_test_predicted, y_predict)

    print("accuracy of the model: {}".format(calc_accuracy_score(y_test, y_test_predicted)))
    true_pos, true_neg, false_pos, false_neg = calc_confusion_matrix(y_test, y_test_predicted)
    print(true_pos, true_neg)
    print(false_pos, false_neg)


def fit_SVM(training_data_dir):
    x_train, x_test, y_train, y_test = proc_CSV_data(fl_dir=training_data_dir)
    # dataframe = pd.read_csv(training_data_dir)
    # dataframe["feature_class"]. \
    #     replace({"ASD": -1.0, "TD": 1.0},
    #             inplace=True)
    # dataframe = dataframe.sample(frac=1)
    # X = dataframe.drop(labels="feature_class", axis=1)
    # pca = PCA(n_components=2)
    # pca = pca.fit(X)
    # X = pca.transform(X)
    # Y = dataframe['feature_class']
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    print("Training SVM")  # debug
    trained_weights = stochastic_gradient_descent(x_train.to_numpy(), y_train.to_numpy())

    print("Testing SVM")  # debug
    test_SVM(weights=trained_weights, x_test=x_test.to_numpy(), y_test=y_test.to_numpy())

    print("Weights are: ", trained_weights)


if __name__ == "__main__":
    # proc_CSV_data("D:/TrainingDataset_YEAR_PROJECT/TrainingSet.csv")
    fit_SVM("D:/TrainingDataset_YEAR_PROJECT/TrainingSet.csv")
