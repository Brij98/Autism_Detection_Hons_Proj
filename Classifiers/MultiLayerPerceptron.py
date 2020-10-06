import json
import os
import time
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

from Classifiers import Utils

# MultilayerPerceptronMdlFl = 'Trained_Classifiers/MultiLayerPerceptronModel.sav'
MultilayerPerceptronMdlFl = "C:/Users/Brijesh Prajapati/Documents/Projects/Autism_Detection_Hons_Proj/Classifiers/" \
                            "Trained_Classifiers/MultiLayerPerceptronModel.sav"


class MultiLayerPerceptron:
    def __init__(self, training_data_dir):
        self.__max_iter = 400
        self.__hidden_layer_size = 16
        self.training_data_dir = training_data_dir
        self.__mlp_classifier = MLPClassifier()
        self.__x_train = None
        self.__x_test = None
        self.__y_train = None
        self.__y_test = None

        # calculating the min max scalar for normalization
        self.__min_max_scalar = Utils.calculate_min_max_np_scalar(pd.read_csv(training_data_dir))

    # training MLP
    # X_train: training feature values
    # X_test: testing feature values
    # y_train: labels for the training values
    # y_test: labels for the testing values
    # save_mdl: will save the weights if true
    # save_loc: if save_mdl true then save the model to the specified loc
    def train_MultiLayerPerceptron(self, X_train, X_test, y_train, y_test, save_mdl=False,
                                   save_loc=MultilayerPerceptronMdlFl, hidden_layer_sizes=16,
                                   max_iter=400):

        x_train_cpy = Utils.normalize_numpy_array(X_train, self.__min_max_scalar)
        x_test_cpy = Utils.normalize_numpy_array(X_test, self.__min_max_scalar)

        self.__x_train = x_train_cpy
        self.__x_test = x_test_cpy
        self.__y_train = y_train
        self.__y_test = y_test

        self.__max_iter = max_iter
        self.__hidden_layer_size = hidden_layer_sizes

        print("Training MultiLayerPerceptron...")
        t0 = time.time()  # test purposes

        self.__mlp_classifier = MLPClassifier(hidden_layer_sizes= self.__hidden_layer_size, activation='relu',
                                              max_iter=self.__max_iter, solver='adam')

        self.__mlp_classifier = self.__mlp_classifier.fit(self.__x_train, self.__y_train)

        t1 = time.time()  # test purposes
        print("Finished Training MultiLayerPerceptron in: ", t1 - t0)

        self.__test_MultiLayerPerceptron()

        if save_mdl:
            self.__save_MultiLayerPerceptron(save_loc)

    def __test_MultiLayerPerceptron(self):
        print("Testing Multi Layer Perceptron")
        mlp_predictions = self.__mlp_classifier.predict(self.__x_test)

        accuracy_score = Utils.calculate_accuracy_score(self.__y_test, mlp_predictions)
        print("Accuracy: ", accuracy_score)

        cf = Utils.calculate_confusion_matrix(self.__y_test, mlp_predictions)
        print("Multi Layer Perceptron Confusion Matrix")
        print(cf[0])
        print(cf[1])

        print(classification_report(self.__y_test, mlp_predictions))

        # write accuracy and confusion matrix to file

        try:
            report_fl_name = "C:/Users/Brijesh Prajapati/Documents/Projects/Autism_Detection_Hons_Proj/Classifiers/" \
                             "Classifier_Reports/mlp_current_report"

            classifier_desc = "MLP hidden_layer_size: " + str(self.__hidden_layer_size) + ", max_iteration: " + \
                              str(self.__max_iter)
            dict_report = {"desc": classifier_desc, "accuracy_score": str(accuracy_score),
                           "tp": str(cf[0][0]), "fn": str(cf[0][1]),
                           "fp": str(cf[1][0]), "tn": str(cf[1][1]),
                           "specificity": str(cf[2][0]), "sensitivity": str(cf[2][1])}

            with open(report_fl_name, 'w') as report_fl:
                report_fl.write(json.dumps(dict_report))
        except Exception as ex:
            print("Exception occurred saving MLP report", ex)

    def predict(self, sample_data, return_lbl=False):
        # normalize the sample data
        # sample_data = Utils.normalize_dataset(sample_data, self.__min_max_scalar)
        sample_data = Utils.normalize_numpy_array(sample_data, self.__min_max_scalar)

        prediction = self.__mlp_classifier.predict([sample_data])

        if return_lbl is True:
            if prediction > 0:
                return 'ASD'
            else:
                return 'Normal'
        else:
            return prediction

    def __save_MultiLayerPerceptron(self, fl_loc=MultilayerPerceptronMdlFl):
        try:
            os.remove(fl_loc)
            print("Deleted previous file")
            joblib.dump(self.__mlp_classifier, fl_loc)
        except Exception as e:
            print("Error saving MultiLayerPerceptron Model", e)

    def load_MultiLayerPerceptron(self, mdl_fl=MultilayerPerceptronMdlFl):
        try:
            self.__mlp_classifier = joblib.load(mdl_fl)
        except Exception as e:
            print("Error loading MultiLayerPerceptron Model", e)


def process_training_data(fl_dir, min_max_scalar=None):
    df = pd.read_csv(fl_dir)

    # replace labels
    df['feature_class'].replace({'ASD': 1.0, 'TD': -1.0}, inplace=True)

    x_train, x_test, y_train, y_test = Utils.train_test_split(df, 0.2)

    # if min_max_scalar is not None:
    #     x_train = Utils.normalize_dataset(x_train, min_max_scalar)
    #     x_test = Utils.normalize_dataset(x_test, min_max_scalar)

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    mlp_classifier = MultiLayerPerceptron(
        training_data_dir="D:/TrainingDataset_YEAR_PROJECT/TrainingSet_All.csv")
    # mlp_classifier.train_MultiLayerPerceptron(save_mdl=False)
    mlp_classifier.test_v2()

    # mlp_classifier.load_MultiLayerPerceptron()
    #
    # data = [[3, 192, 64, 5692, 2846, 55.5706155869889, 32.2860284249144]]
    # df = pd.DataFrame(data, columns=["fixpoint_count", "total_duration", "mean_duration", "total_scanpath_len",
    #                                  "mean_scanpath_len", "mean_dist_centre", "mean_dist_mean_coord"])
    #
    # print(mlp_classifier.predict(df))
