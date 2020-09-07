import time

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
        self.training_data_dir = training_data_dir
        self.__mlp_classifier = MLPClassifier(hidden_layer_sizes=9, activation='relu', max_iter=400, solver='adam')
        self.__x_train = None
        self.__x_test = None
        self.__y_train = None
        self.__y_test = None

        # calculating the min max scalar for normalization
        self.__min_max_scalar = Utils.calculate_min_max_scalar(pd.read_csv(training_data_dir))

    def train_MultiLayerPerceptron(self, save_mdl=True):
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = process_training_data(self.training_data_dir,
                                                                                             self.__min_max_scalar)
        print("Training MultiLayerPerceptron...")
        t0 = time.time()  # test purposes

        self.__mlp_classifier = self.__mlp_classifier.fit(self.__x_train, self.__y_train)

        t1 = time.time()  # test purposes
        print("Finished Training Random Forest Classifier in: ", t1 - t0)

        self.__test_MultiLayerPerceptron()

        if save_mdl is True:
            self.__save_MultiLayerPerceptron()

    def __test_MultiLayerPerceptron(self):
        print("Testing Multi Layer Perceptron")
        mlp_predictions = self.__mlp_classifier.predict(self.__x_test)

        accuracy_score = Utils.calculate_accuracy_score(self.__y_test, mlp_predictions)
        print("Accuracy: ", accuracy_score)

        cf = Utils.calculate_confusion_matrix(self.__y_test, mlp_predictions)
        print(cf[0])
        print(cf[1])

        print(classification_report(self.__y_test, mlp_predictions))

    def predict(self, sample_data):
        # normalize the sample data
        sample_data = Utils.normalize_dataset(sample_data, self.__min_max_scalar)

        prediction = self.__mlp_classifier.predict(sample_data)

        if prediction > 0:
            return 'ASD'
        else:
            return 'Normal'

    def __save_MultiLayerPerceptron(self):
        try:
            joblib.dump(self.__mlp_classifier, MultilayerPerceptronMdlFl)
        except Exception as e:
            print("Error saving MultiLayerPerceptron Model", e)

    def load_MultiLayerPerceptron(self):
        try:
            self.__mlp_classifier = joblib.load(MultilayerPerceptronMdlFl)
        except Exception as e:
            print("Error loading MultiLayerPerceptron Model", e)


def process_training_data(fl_dir, min_max_scalar=None):
    df = pd.read_csv(fl_dir)

    # replace labels
    df['feature_class'].replace({'ASD': 1, 'TD': 0}, inplace=True)

    x_train, x_test, y_train, y_test = Utils.train_test_split(df, 0.2)

    if min_max_scalar is not None:
        x_train = Utils.normalize_dataset(x_train, min_max_scalar)
        x_test = Utils.normalize_dataset(x_test, min_max_scalar)

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    mlp_classifier = MultiLayerPerceptron(training_data_dir="D:/TrainingDataset_YEAR_PROJECT/TrainingSet.csv")
    # mlp_classifier.train_MultiLayerPerceptron(save_mdl=False)

    mlp_classifier.load_MultiLayerPerceptron()

    data = [[3, 192, 64, 5692, 2846, 55.5706155869889, 32.2860284249144]]
    df = pd.DataFrame(data, columns=["fixpoint_count", "total_duration", "mean_duration", "total_scanpath_len",
                                     "mean_scanpath_len", "mean_dist_centre", "mean_dist_mean_coord"])

    print(mlp_classifier.predict(df))
