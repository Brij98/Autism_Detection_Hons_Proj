import json

from Classifiers.SupportVectorMachine import SupportVectorMachine
from Classifiers.RandomForest import RandomForest
from Classifiers.MultiLayerPerceptron import MultiLayerPerceptron
from Classifiers.DecisionTree import most_common_label
from Classifiers.AdaBoostClassifier import AdaBoostClassifier

import numpy as np

from Classifiers import Utils

import pandas as pd

# Training_Data_Dir = "D:/TrainingDataset_YEAR_PROJECT/TrainingSet_All.csv"
Training_Data_Dir = "C:/Users/Brijesh Prajapati/Documents/Projects/Autism_Detection_Hons_Proj/Features/" \
                    "Extracted_Feature_Files/TrainingSet_All.csv"


class MainClassifier:
    def __init__(self):
        # checking if the models are trained/ or are loaded
        self.__models_loaded = True

        # SVM
        self.svm = SupportVectorMachine(Training_Data_Dir)

        # Random Forest
        self.random_forest = RandomForest()

        # Multi Layer Perceptron (Neural Network)
        self.multi_layer_perceptron = MultiLayerPerceptron(Training_Data_Dir)

        # Adaboost
        self.adaboost = AdaBoostClassifier()

        # self.__x_train, self.__x_test, self.__y_train, self.__y_test = None
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = prepare_data_for_training(Training_Data_Dir)

    def train_classifiers(self, svm_param_dict=None, rf_param_dict=None, mlp_param_dict=None,
                          adab_param_dict=None):

        self.__x_train, self.__x_test, self.__y_train, self.__y_test = prepare_data_for_training(Training_Data_Dir)

        save_mdl = True

        # ******************************************************************************Setting Up SVM for training
        x_temp, y_temp = Utils.bootstrap_samples(self.__x_train, self.__y_train)  # bootstrap samples
        if svm_param_dict is not None:
            self.svm.train_support_vector_machine(x_temp, self.__x_test, y_temp, self.__y_test, save_mdl=save_mdl,
                                                  **svm_param_dict)
        else:
            self.svm.train_support_vector_machine(x_temp, self.__x_test, y_temp, self.__y_test)

        # ******************************************************************************Setting Up RF for training
        x_temp, y_temp = Utils.bootstrap_samples(self.__x_train, self.__y_train)  # bootstrap samples
        if rf_param_dict is not None:
            self.random_forest.train_random_forest(x_temp, self.__x_test, y_temp, self.__y_test, save_mdl=save_mdl,
                                                   **rf_param_dict)
        else:
            self.random_forest.train_random_forest(x_temp, self.__x_test, y_temp, self.__y_test, save_mdl=save_mdl)

        # ******************************************************************************Setting Up MLP for training
        x_temp, y_temp = Utils.bootstrap_samples(self.__x_train, self.__y_train)
        if mlp_param_dict is not None:
            self.multi_layer_perceptron.train_MultiLayerPerceptron(x_temp, self.__x_test, y_temp, self.__y_test,
                                                                   save_mdl=save_mdl, **mlp_param_dict)
        else:
            self.multi_layer_perceptron.train_MultiLayerPerceptron(x_temp, self.__x_test, y_temp, self.__y_test,
                                                                   save_mdl=save_mdl)

        # ******************************************************************************Setting Up Adaboost for training
        x_temp, y_temp = Utils.bootstrap_samples(self.__x_train, self.__y_train)
        if adab_param_dict is not None:
            self.adaboost.train_adaboost(x_temp, self.__x_test, y_temp, self.__y_test, save_classifiers=save_mdl,
                                         **adab_param_dict)
        else:
            self.adaboost.train_adaboost(x_temp, self.__x_test, y_temp, self.__y_test, save_classifiers=save_mdl)

        # self.__models_loaded = True

        # self.test_main_classifier()

        return 0

    def __load_classifiers(self):
        self.svm.load_trained_weights()
        self.random_forest.load_decision_trees()
        self.multi_layer_perceptron.load_MultiLayerPerceptron()
        self.adaboost.load_adaboost_classifier()

        # self.__models_loaded = True

    def predict_sample(self, sample_data):
        # if self.__models_loaded is False:
        #     self.__load_classifiers()

        self.__load_classifiers()

        svm_p = self.svm.predict_sample(sample_data)
        print("SVM predicts: ", float(svm_p[0]))

        rf_p = self.random_forest.predict(sample_data)
        print("random forest predicts: ", rf_p)

        mlp_p = self.multi_layer_perceptron.predict(sample_data.to_numpy()[0])
        print("multi layer perceptron predicts: ", mlp_p)

        adb_p = self.adaboost.predict_sample_adaboost(sample_data)
        print("adaboost predicts: ", adb_p)

        # predictions = [svm_p, rf_p, mlp_p, adb_p]
        predictions = [float(svm_p[0]), rf_p[0], mlp_p[0], adb_p[0]]

        dict_to_ret = {
            "Support Vector Machine Prediction": float(svm_p),
            "Random Forest Prediction": rf_p[0],
            "Multi Layer Perceptron Prediction": mlp_p[0],
            "AdaBoost Prediction": adb_p[0],
            "Final Prediction": most_common_label(predictions)
        }

        return dict_to_ret, most_common_label(predictions)

    def test_main_classifier(self):
        self.__load_classifiers()

        y_pred = []
        for test_data in self.__x_test:
            svm_p = self.svm.predict_sample([test_data])
            rf_p = self.random_forest.predict([test_data])
            mlp_p = self.multi_layer_perceptron.predict(test_data)
            adb_p = self.adaboost.predict_sample_adaboost(np.array([test_data]))

            predictions = [float(svm_p), rf_p[0], mlp_p[0], adb_p[0]]

            y_pred.append(most_common_label(predictions))

        accuracy_score = Utils.calculate_accuracy_score(self.__y_test, y_pred)
        print("Accuracy of main model: ", accuracy_score)

        confusion_mat = Utils.calculate_confusion_matrix(self.__y_test, y_pred)
        print("Main Classifier Confusion Matrix")
        print(confusion_mat[0])
        print(confusion_mat[1])

        # write accuracy and confusion matrix to file

        try:
            report_fl_name = "C:/Users/Brijesh Prajapati/Documents/Projects/Autism_Detection_Hons_Proj/Classifiers/" \
                             "Classifier_Reports/mainmodel_current_report.json"

            classifier_desc = "Main Model"
            dict_report = {"desc": classifier_desc, "accuracy_score": str(accuracy_score),
                           "tp": str(confusion_mat[0][0]), "fn": str(confusion_mat[0][1]),
                           "fp": str(confusion_mat[1][0]), "tn": str(confusion_mat[1][1]),
                           "specificity": str(confusion_mat[2][0]), "sensitivity": str(confusion_mat[2][1])}

            with open(report_fl_name, 'w') as report_fl:
                report_fl.write(json.dumps(dict_report))
        except Exception as ex:
            print("Exception occurred saving Main Model report", ex)


def prepare_data_for_training(file_dir):
    dataframe = pd.read_csv(file_dir)

    # replacing the labels
    dataframe['feature_class']. \
        replace({'ASD': 1.0, 'TD': -1.0}, inplace=True)

    # splitting data into train and test
    x_train, x_test, y_train, y_test = Utils.train_test_split(dataframe=dataframe, test_size=0.2)

    # convert to dataframe to numpy array and return
    return x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()


if __name__ == "__main__":
    mc = MainClassifier()
    # mc.train_classifiers()
    mc.test_main_classifier()

    # data = [[3, 192, 64, 5692, 2846, 55.5706155869889, 32.2860284249144]]
    # data = [[12, 2564, 213.666666666666, 29597, 2690.63636363636, 176.05935069273, 132.591777456996]]
    # df = pd.DataFrame(data, columns=["fixpoint_count", "total_duration", "mean_duration", "total_scanpath_len",
    #                                  "mean_scanpath_len", "mean_dist_centre", "mean_dist_mean_coord"])
    feature_values = [[105, 20, 20, 52.083333333333336, 625, 194999.0, 16249.916666666666, 147, 9.715030072847767,
                       0.4139319990291013, 12, 3938.0, 328.1666666666667, 61168.0, 5097.333333333333,
                       247.89632236438797, 214.19275070700743]]

    df = pd.DataFrame(feature_values,
                      columns=["first_saliency_fixation", "first_above_0.75_max_rank", "first_above_0.9_max_rank",
                               "saliency_value_mean", "saliency_value_sum", "weighted_duration_sum",
                               "weighted_duration_mean", "max_saliency_value", "relative_entropy",
                               "normalized_saliency_scanpath",
                               "fixpoint_count", "total_duration", "mean_duration", "total_scanpath_len",
                               "mean_scanpath_len", "mean_dist_centre", "mean_dist_mean_coord"])

    prediction = mc.predict_sample(df)[1]
    print(prediction)
