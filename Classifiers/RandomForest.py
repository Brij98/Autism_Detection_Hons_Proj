import concurrent.futures
import json
import os
import time
import pickle
import traceback

import numpy as np
import pandas as pd
# from sklearn.metrics import confusion_matrix, classification_report

from Classifiers import Utils
from Classifiers.DecisionTree import DecisionTree, most_common_label

# RandomForestMdlFile = 'Trained_Classifiers/RandomForestClassifierModel'
RandomForestMdlFile = "C:/Users/Brijesh Prajapati/Documents/Projects/Autism_Detection_Hons_Proj/Classifiers/" \
                      "Trained_Classifiers/RandomForestClassifierModel"


class RandomForest:
    def __init__(self):
        self.__num_tree = 10
        self.__max_depth = 8
        self.__arr_trees = []
        self.__x_train = None
        self.__x_test = None
        self.__y_train = None
        self.__y_test = None

        # finding the min max scalar
        # self.min_max_scalar = \
        #     Utils.calculate_min_max_scalar(pd.read_csv("D:/TrainingDataset_YEAR_PROJECT/TrainingSet.csv"))

    def train_random_forest(self, x_train, x_test, y_train, y_test, save_mdl=False, save_loc=RandomForestMdlFile,
                            num_trees=10,  max_depth=8):

        # setting attributes
        self.__num_tree = num_trees
        self.__max_depth = max_depth

        print("Training Random Forest...")  # debug

        self.__x_train, self.__x_test, self.__y_train, self.__y_test = x_train, x_test, y_train, y_test

        self.__arr_trees = []
        t0 = time.time()  # test purposes

        # single thread approach
        # for k in range(self.num_tree):
        #     print("training_decision_tree")
        #     tree = DecisionTree(min_samples_split=self.min_sample_split, max_depth=self.max_depth,
        #                         num_features=self.num_features)
        #     x_sample, y_sample = bootstrap_samples(self.x_train, self.y_train)
        #     tree.train_decision_tree(x_sample, y_sample)
        #     self.arr_trees.append(tree)

        # thread pool approach
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            decision_trees = {executor.submit(trained_decision_tree,
                                              self.__max_depth, self.__x_train, self.__y_train): i
                              for i in range(self.__num_tree)}

            for future in concurrent.futures.as_completed(decision_trees):
                try:
                    self.__arr_trees.append(future.result())
                except Exception as exc:
                    print("Error occurred: ", exc)

        t1 = time.time()  # test purposes
        print("Finished Training Random Forest Classifier in: ", t1 - t0)

        self.__test_random_forest()

        if save_mdl:
            self.__save_decision_trees(save_loc)

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

        # write accuracy and confusion matrix to file

        try:
            report_fl_name = "C:/Users/Brijesh Prajapati/Documents/Projects/Autism_Detection_Hons_Proj/Classifiers/" \
                             "Classifier_Reports/rf_current_report"

            classifier_desc = "RF number of trees: " + str(self.__num_tree) + ", max depth: " + \
                              str(self.__max_depth)
            dict_report = {"desc": classifier_desc, "accuracy_score": str(accuracy_score),
                           "tp": str(cf[0][0]), "fn": str(cf[0][1]),
                           "fp": str(cf[1][0]), "tn": str(cf[1][1]),
                           "specificity": str(cf[2][0]), "sensitivity": str(cf[2][1])}

            with open(report_fl_name, 'w') as report_fl:
                report_fl.write(json.dumps(dict_report))
        except Exception as ex:
            print("Exception occurred saving random forest report", ex)

    # given input features classify the features and return a decision
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
            try:
                os.remove(fl_name)
                print("Deleted previous file")
                with open(fl_name, 'wb') as output:
                    for tree in self.__arr_trees:
                        pickle.dump(tree, output, pickle.HIGHEST_PROTOCOL)
            except Exception as ex:
                print("Exception occurred saving decision tree", ex)
                traceback.print_exc()
        else:
            print("Train the Random Forest Classifier before saving")

    # load the decision trees
    def load_decision_trees(self, fl_name=RandomForestMdlFile):
        self.__arr_trees = []

        try:
            with open(fl_name, 'rb') as input_tree:
                while True:
                    try:
                        self.__arr_trees.append(pickle.load(input_tree))
                    except EOFError:
                        break
        except Exception as e:
            print("Error occurred while opening and reading file")
            print(e)
            traceback.print_exc()

        # try:
        #     with open(fl_name, 'rb') as input_tree:
        #         for i in range(num_trees):
        #             self.__arr_trees.append(pickle.load(input_tree))
        #
        # except Exception as e:
        #     print("Error occurred while loading the decision trees", e)


# used for training a single decision tree.
def trained_decision_tree(max_depth, x, y):
    # Create decision tree
    # print("Called trained_decision_tree")  # debug
    dcsn_tree = DecisionTree(max_depth=max_depth)
    x_sample, y_sample = Utils.bootstrap_samples(x, y)  # bootstrapping samples
    dcsn_tree.train_decision_tree(x_sample, y_sample)

    return dcsn_tree


# bootstrapping samples
# def bootstrap_samples(x, y):
#     num_samples = x.shape[0]
#
#     # randomly select indices and allow for same index to occur more than once
#     indices = np.random.choice(num_samples, size=num_samples, replace=True)
#     return x[indices], y[indices]


# renaming feature labels to 1 and 0 and converting dataframe into a numpy array
def process_training_data(fl_dir):
    df = pd.read_csv(fl_dir)

    # finding min max scalar
    # min_max_scalar = Utils.calculate_min_max_scalar(pd.read_csv(fl_dir))

    # replace labels
    df['feature_class'].replace({'ASD': 1.0, 'TD': -1.0}, inplace=True)

    X_train, X_test, y_train, y_test = Utils.train_test_split(df, 0.2)

    # X_train = Utils.normalize_dataset(X_train, min_max_scalar)
    # X_test = Utils.normalize_dataset(X_test, min_max_scalar)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    rf = RandomForest()
    # rf.train_random_forest("D:/TrainingDataset_YEAR_PROJECT/TrainingSet_Saliency_2.csv")
    # rf.test_v2("D:/TrainingDataset_YEAR_PROJECT/TrainingSet_All.csv")

    # data = [[3, 192, 64, 5692, 2846, 55.5706155869889, 32.2860284249144]]
    # df = pd.DataFrame(data, columns=["fixpoint_count", "total_duration", "mean_duration", "total_scanpath_len",
    #                                  "mean_scanpath_len", "mean_dist_centre", "mean_dist_mean_coord"])
    #
    # ret_val = rf.predict(df, True)
    # print(ret_val)
