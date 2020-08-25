import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from Classifiers import Utils, SupportVectorMachine
from Classifiers.DecisionTree import DecisionTree, most_common_label


class RandomForest:
    def __init__(self, num_trees=100, min_samples_split=2, max_depth=10, num_features=None):
        self.num_tree = num_trees
        self.min_sample_split = min_samples_split
        self.max_depth = max_depth
        self.num_features = num_features
        self.arr_tree = []

    def fit(self, x, y):
        self.arr_tree = []
        for k in range(self.num_tree):
            tree = DecisionTree(min_samples_split=self.min_sample_split, max_depth=self.max_depth,
                                num_features=self.num_features)
            x_sample, y_sample = bootstrap_samples(x, y)
            tree.fit(x_sample, y_sample)
            self.arr_tree.append(tree)

    def predict(self, x):
        tree_predictions = []
        for tree in self.arr_tree:
            tree_predictions.append(tree.predict(x))
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)
        y_pred = []
        for i in tree_predictions:
            y_pred.append(most_common_label(i))

        return np.array(y_pred)


def bootstrap_samples(x, y):
    num_samples = x.shape[0]
    indices = np.random.choice(num_samples, size=num_samples, replace=True)
    return x[indices], y[indices]


if __name__ == '__main__':
    df = pd.read_csv("D:/TrainingDataset_YEAR_PROJECT/TrainingSet.csv")

    # replace labels
    df['feature_class'].replace({'ASD': 0, 'TD': 1}, inplace=True)

    X_train, X_test, y_train, y_test = Utils.train_test_split(df, 0.2)

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    decision_tree = RandomForest(num_trees=18, max_depth=10)
    decision_tree.fit(X_train, y_train)

    y_pred = decision_tree.predict(X_test)

    accuracy_score = Utils.calculate_accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy_score)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # print(Y_test)
    # print(y_pred)

    print(accuracy_score(y_test, y_pred))
