from Classifiers.SupportVectorMachine import SupportVectorMachine
from Classifiers.RandomForest import RandomForest
from Classifiers.MultiLayerPerceptron import MultiLayerPerceptron
from Classifiers.DecisionTree import most_common_label

import pandas as pd

Training_Data_Dir = "D:/TrainingDataset_YEAR_PROJECT/TrainingSet.csv"


class MainClassifier:
    def __init__(self):
        # checking if the models are trained/ or are loaded
        self.__models_loaded = False

        # SVM
        self.svm = SupportVectorMachine(Training_Data_Dir)

        # Random Forest
        self.random_forest = RandomForest(num_trees=5, min_samples_split=2, max_depth=10)

        # Multi Layer Perceptron (Neural Network)
        self.multi_layer_perceptron = MultiLayerPerceptron(Training_Data_Dir)

    def train_classifiers(self):
        self.svm.train_support_vector_machine(save_mdl=True)
        self.random_forest.train_random_forest(Training_Data_Dir, save_mdl=True)
        self.multi_layer_perceptron.train_MultiLayerPerceptron(save_mdl=True)

        self.__models_loaded = True

    def __load_classifiers(self):
        self.svm.load_trained_weights()
        self.random_forest.load_decision_trees(5)
        self.multi_layer_perceptron.load_MultiLayerPerceptron()

        self.__models_loaded = True

    def predict_sample(self, sample_data):
        if self.__models_loaded is False:
            self.__load_classifiers()

        svm_p = self.svm.predict_sample(sample_data, True)
        print("SVM predicts: ", svm_p)

        rf_p = self.random_forest.predict(sample_data, True)
        print("random forest predicts: ", rf_p)

        mlp_p = self.multi_layer_perceptron.predict(sample_data)
        print("multi layer perceptron predicts: ", mlp_p)

        predictions = [svm_p, rf_p, mlp_p]

        return most_common_label(predictions)


if __name__ == "__main__":
    mc = MainClassifier()
    # mc.train_classifiers()

    # data = [[3, 192, 64, 5692, 2846, 55.5706155869889, 32.2860284249144]]
    data = [[12, 2564, 213.666666666666, 29597, 2690.63636363636, 176.05935069273, 132.591777456996]]
    df = pd.DataFrame(data, columns=["fixpoint_count", "total_duration", "mean_duration", "total_scanpath_len",
                                     "mean_scanpath_len", "mean_dist_centre", "mean_dist_mean_coord"])

    prediction = mc.predict_sample(df)
    print(prediction)
