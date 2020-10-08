import json
import os
import traceback

import numpy as np
import pandas as pd
from Classifiers import Utils
import pickle

AdaBoostMdlFl = "C:/Users/Brijesh Prajapati/Documents/Projects/Autism_Detection_Hons_Proj/Classifiers/" \
                      "Trained_Classifiers/AdaBoostClassifierModel"


class AdaBoostClassifier:
    def __init__(self):
        self.__num_classifiers = 2
        self.__classifiers = []
        # self.__training_data_dir = training_data_dir
        self.__x_train = None
        self.__x_test = None
        self.__y_train = None
        self.__y_test = None

    def train_adaboost(self, x_train, x_test, y_train, y_test, save_classifiers=False, save_loc=AdaBoostMdlFl,
                       num_classifiers=5):

        print("Training Adaboost Classifier")   # debug

        self.__num_classifiers = num_classifiers
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = x_train, x_test, y_train, y_test

        num_samples, num_features = self.__x_train.shape

        # initializing weights array
        weights = np.zeros(num_samples)
        # initializing weights to 1/N
        for i in range(num_samples):
            weights[i] = 1 / num_samples

        for j in range(self.__num_classifiers):
            classifier = DecisionStump()

            # searching to find the best feature and best feature value to split
            min_error_val = float('inf')

            # iterate over each feature
            for idx in range(num_features):
                col = self.__x_train[:, idx]
                # thresh = np.unique(col)

                # iterate over each value in the col
                for val in col:
                    polarity = 1

                    # performing prediction on each value of the col
                    predictions = np.ones(num_samples)
                    predictions[col < val] = -1

                    # get the weights of samples that are wrongly classified i.e samples that do not match
                    # with predictions
                    wrong_classifications = weights[self.__y_train != predictions]

                    # calculating the error of wrongly classified weights by summing the weights of the wrongly
                    # classifier values
                    error_val = sum(wrong_classifications)

                    # flip error and polarity if error_val > 0.5
                    if error_val > 0.5:
                        error_val = 1 - error_val
                        polarity = -1

                    # checking if error is less than min error to set up the stub (weak classifier)
                    if error_val < min_error_val:
                        min_error_val = error_val
                        classifier.polarity = polarity
                        classifier.split_threshold = val
                        classifier.feature_idx = idx

            # calculating the alpha value 0.5 * log ((1 - min_error) / min_error)
            EPS = 1e-10
            classifier.alpha_val = 0.5 * np.log((1 - min_error_val) / (min_error_val + EPS))

            predictions = classifier.predict(self.__x_train)

            # updating weights. new weights =(prev weight exp(-alpha * label * predictions)) / sum(weights)
            weights *= np.exp(-classifier.alpha_val * self.__y_train * predictions)
            # normalizing the weights
            weights /= np.sum(weights)

            self.__classifiers.append(classifier)

        # testing adaboost
        self.__test_adaboost()

        if save_classifiers:
            self.__save_adaboost_classifiers()

    def predict_sample_adaboost(self, X, return_lbl=False):
        # predicted_label = sign(SUM(alpha * prediction))
        if self.__classifiers is None:
            try:
                self.load_adaboost_classifier(AdaBoostMdlFl)
            except Exception as ex:
                print("Error loading weights")

        if self.__classifiers is not None:
            classifier_predictions = []

            if isinstance(X, pd.DataFrame):
                X = X.to_numpy()

            for classifier in self.__classifiers:
                classifier_predictions.append(classifier.alpha_val * classifier.predict(X))
            prediction = np.sum(classifier_predictions, axis=0)
            prediction = np.sign(prediction)

            if return_lbl:
                if prediction < 0:
                    return 'ASD'
                else:
                    return 'TD'

            return prediction
        else:
            return 0

    # testing random forest classifier
    def __test_adaboost(self):
        print("Testing AdaBoost")

        y_predictions = np.ones(len(self.__x_test))
        for test, i in zip(self.__x_test, range(len(self.__x_test))):
            test = np.reshape(test, (-1, len(test)))
            y_predictions[i] = self.predict_sample_adaboost(test)

        accuracy_score = Utils.calculate_accuracy_score(self.__y_test, y_predictions)
        print("Accuracy: ", accuracy_score)

        cf = Utils.calculate_confusion_matrix(self.__y_test, y_predictions)
        print("Adaboost Matrix")
        print(cf[0])
        print(cf[1])

        # write accuracy and confusion matrix to file

        try:
            report_fl_name = "C:/Users/Brijesh Prajapati/Documents/Projects/Autism_Detection_Hons_Proj/Classifiers/" \
                             "Classifier_Reports/adaboost_current_report.json"

            classifier_desc = "Adaboost number of stumps: " + str(self.__num_classifiers)
            dict_report = {"desc": classifier_desc, "accuracy_score": str(accuracy_score),
                           "tp": str(cf[0][0]), "fn": str(cf[0][1]),
                           "fp": str(cf[1][0]), "tn": str(cf[1][1]),
                           "specificity": str(cf[2][0]), "sensitivity": str(cf[2][1])}

            with open(report_fl_name, 'w') as report_fl:
                report_fl.write(json.dumps(dict_report))
        except Exception as ex:
            print("Exception occurred saving adaboost report", ex)

    def __save_adaboost_classifiers(self, fl_name=AdaBoostMdlFl):
        if self.__classifiers is not None:
            try:
                os.remove(fl_name)
                print("Deleted previous file")
                with open(fl_name, 'wb') as output:
                    for classifier in self.__classifiers:
                        pickle.dump(classifier, output, pickle.HIGHEST_PROTOCOL)
                # return True
            except Exception as e:
                print("Exception saving the adaboost classifier")
                print("Exception", e)
                traceback.print_exc()
                # return False
        else:
            print("Classifiers list is empty. Train the classifiers first.")
            # return False

    def load_adaboost_classifier(self, fl_name=AdaBoostMdlFl):
        self.__classifiers = []
        try:
            with open(fl_name, 'rb') as input_classifier:
                while True:
                    try:
                        self.__classifiers.append(pickle.load(input_classifier))
                    except EOFError:
                        break

        except Exception as e:
            print("Error occurred while opening and reading file")
            print(e)
            traceback.print_exc()


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


# decision tree with a depth of 1
class DecisionStump:
    def __init__(self):
        self.polarity = 1  # is used to change '>' '<' signs
        self.feature_idx = None
        self.split_threshold = None
        self.alpha_val = None

    # accepts the sample set to predict
    def predict(self, x):
        n_samples = x.shape[0]
        col = x[:, self.feature_idx]

        # array of positive class
        x_prediction = np.ones(n_samples)
        if self.polarity == 1:
            # change the class to -ve where values are less than the split_threshold
            x_prediction[col < self.split_threshold] = -1
        else:
            # change the class to -ve where values are greater than the split_threshold
            x_prediction[col > self.split_threshold] = -1

        return x_prediction


if __name__ == "__main__":
    adaboost = AdaBoostClassifier("D:/TrainingDataset_YEAR_PROJECT/TrainingSet.csv", num_classifiers=12)
    adaboost.train_adaboost(save_classifiers=True)
    # data = [[12, 2564, 213.666666666666, 29597, 2690.63636363636, 176.05935069273, 132.591777456996]]
    # df = pd.DataFrame(data, columns=["fixpoint_count", "total_duration", "mean_duration", "total_scanpath_len",
    #                                  "mean_scanpath_len", "mean_dist_centre", "mean_dist_mean_coord"])
    #
    # ret_val = adaboost.predict_sample_adaboost(df, True)
    # print(ret_val)
