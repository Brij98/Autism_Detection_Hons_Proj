import concurrent.futures
import time
import pickle
import numpy as np
import pandas as pd
# from sklearn.metrics import confusion_matrix, classification_report

from Classifiers import Utils
from Classifiers.DecisionTree import DecisionTree, most_common_label

# RandomForestMdlFile = 'Trained_Classifiers/RandomForestClassifierModel'
RandomForestMdlFile = "C:/Users/Brijesh Prajapati/Documents/Projects/Autism_Detection_Hons_Proj/Classifiers/" \
                      "Trained_Classifiers/RandomForestClassifierModel"


class RandomForest:
    def __init__(self, num_trees=10, min_samples_split=2, max_depth=8, num_features=None):
        self.num_tree = num_trees
        self.__min_sample_split = min_samples_split
        self.__max_depth = max_depth
        self.__num_features = num_features
        self.__arr_trees = []
        self.__x_train = None
        self.__x_test = None
        self.__y_train = None
        self.__y_test = None

        # finding the min max scalar
        # self.min_max_scalar = \
        #     Utils.calculate_min_max_scalar(pd.read_csv("D:/TrainingDataset_YEAR_PROJECT/TrainingSet.csv"))

    def train_random_forest(self, training_data_dir, save_mdl=False):
        print("Training Random Forest...")  # debug

        self.__x_train, self.__x_test, self.__y_train, self.__y_test = process_training_data(training_data_dir)

        self.__arr_trees = []
        t0 = time.time()  # test purposes

        # linear thread approach
        # for k in range(self.num_tree):
        #     print("training_decision_tree")
        #     tree = DecisionTree(min_samples_split=self.min_sample_split, max_depth=self.max_depth,
        #                         num_features=self.num_features)
        #     x_sample, y_sample = bootstrap_samples(self.x_train, self.y_train)
        #     tree.train_decision_tree(x_sample, y_sample)
        #     self.arr_trees.append(tree)

        # thread pool approach
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            decision_trees = {executor.submit(trained_decision_tree, self.__min_sample_split,
                                              self.__max_depth, self.__num_features, self.__x_train, self.__y_train): i
                              for i in range(self.num_tree)}

            for future in concurrent.futures.as_completed(decision_trees):
                try:
                    self.__arr_trees.append(future.result())
                except Exception as exc:
                    print("Error occurred: ", exc)

        t1 = time.time()  # test purposes
        print("Finished Training Random Forest Classifier in: ", t1 - t0)

        self.__test_random_forest()

        if save_mdl is True:
            self.__save_decision_trees()

    # testing random forest classifier
    def __test_random_forest(self):
        print("Testing Random Forest Classifier")
        y_predictions = self.predict(self.__x_test)

        accuracy_score = Utils.calculate_accuracy_score(self.__y_test, y_predictions)
        print("Accuracy: ", accuracy_score)

        cf = Utils.calculate_confusion_matrix(self.__y_test, y_predictions)
        print("Random Forest Confusion Matrix")
        print(cf[0])
        print(cf[1])

    def predict(self, x, return_lbl=False):
        tree_predictions = []

        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()

        #  [[1,0,0,1,1,0], [1,0,0,0,1,0], ..., [0,0,1,1,0,0]]
        for tree in self.__arr_trees:
            tree_predictions.append(tree.predict(x))

        # [[1,1,0], [0,0,0], ..., [0,0,0]]
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)

        if return_lbl is False:
            y_predictions = []
            for i in tree_predictions:
                y_predictions.append(most_common_label(i))
            return np.array(y_predictions)
        else:
            if most_common_label(tree_predictions[0]) > 0:
                return 'ASD'
            else:
                return 'Normal'

    # save the decision trees
    def __save_decision_trees(self, fl_name=RandomForestMdlFile):
        if self.__arr_trees is not None:
            with open(fl_name, 'wb') as output:
                for tree in self.__arr_trees:
                    pickle.dump(tree, output, pickle.HIGHEST_PROTOCOL)
        else:
            print("Train the Random Forest Classifier before saving")

    # load the decision trees
    # num_trees should be the same as the number of trees saved
    def load_decision_trees(self, num_trees, fl_name=RandomForestMdlFile):
        self.__arr_trees = []

        try:
            with open(fl_name, 'rb') as input_tree:
                for i in range(num_trees):
                    self.__arr_trees.append(pickle.load(input_tree))

        except Exception as e:
            print("Error occurred while loading the decision trees", e)


def trained_decision_tree(min_sample_split, max_depth, num_features, x, y):
    # Create decision tree
    # print("Called trained_decision_tree")  # debug
    dcsn_tree = DecisionTree(min_samples_split=min_sample_split, max_depth=max_depth, num_features=num_features)
    x_sample, y_sample = bootstrap_samples(x, y)  # bootstrapping samples
    dcsn_tree.train_decision_tree(x_sample, y_sample)

    return dcsn_tree


# bootstrapping samples
def bootstrap_samples(x, y):
    num_samples = x.shape[0]
    indices = np.random.choice(num_samples, size=num_samples, replace=True)
    return x[indices], y[indices]


def process_training_data(fl_dir):
    df = pd.read_csv(fl_dir)

    # finding min max scalar
    # min_max_scalar = Utils.calculate_min_max_scalar(pd.read_csv(fl_dir))

    # replace labels
    df['feature_class'].replace({'ASD': 1, 'TD': 0}, inplace=True)

    X_train, X_test, y_train, y_test = Utils.train_test_split(df, 0.2)

    # X_train = Utils.normalize_dataset(X_train, min_max_scalar)
    # X_test = Utils.normalize_dataset(X_test, min_max_scalar)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    rf = RandomForest(num_trees=5, max_depth=10)
    rf.train_random_forest("D:/TrainingDataset_YEAR_PROJECT/TrainingSet.csv")

    data = [[3, 192, 64, 5692, 2846, 55.5706155869889, 32.2860284249144]]
    df = pd.DataFrame(data, columns=["fixpoint_count", "total_duration", "mean_duration", "total_scanpath_len",
                                     "mean_scanpath_len", "mean_dist_centre", "mean_dist_mean_coord"])

    ret_val = rf.predict(df, True)
    print(ret_val)
