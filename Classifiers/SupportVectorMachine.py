import traceback

import numpy as np
import pandas as pd

from Classifiers import Utils

Learning_rate = 0.000001
Reg_strength = 10000  # C
# Weights_File = 'Trained_Classifiers/SVM_Weights'
Weights_File = "C:/Users/Brijesh Prajapati/Documents/Projects/Autism_Detection_Hons_Proj/Classifiers/" \
               "Trained_Classifiers/SVM_Weights"


#   SVM
#   1/2 * ||W||^2 + C(1/N * sum(max(0, 1 - Yi * (W * Xi +b))))

class SupportVectorMachine:

    def __init__(self, training_data_dir):
        self.training_data_dir = training_data_dir
        self.__trained_weights = None
        self.__x_train = None
        self.__x_test = None
        self.__y_train = None
        self.__y_test = None

        # finding the min max scalar
        self.__min_max_scalar = Utils.calculate_min_max_scalar(pd.read_csv(training_data_dir))

    def train_support_vector_machine(self, save_mdl=False):
        print("Training SupportVectorMachine...")
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = process_training_data(self.training_data_dir,
                                                                                             self.__min_max_scalar)
        self.__trained_weights = stochastic_gradient_descent(self.__x_train, self.__y_train)

        print(self.__trained_weights)

        print("Finished Training SupportVectorMachine")

        self.__test_support_vector_machine()

        if save_mdl is True:
            self.__save_trained_weights(Weights_File)

    def __test_support_vector_machine(self):
        if self.__trained_weights is not None and self.__x_test is not None and self.__y_test is not None:

            print("Testing SupportVectorMachine")

            y_test_predicted = np.array([])
            # print(type(self.x_test))
            # print(type(self.y_test))

            # for i in range(self.x_test.shape[0]):
            for r in self.__x_test:
                y_predict = np.sign(np.dot(r, self.__trained_weights))
                y_test_predicted = np.append(y_test_predicted, y_predict)

            print("accuracy of the model: {}".format(Utils.calculate_accuracy_score(self.__y_test, y_test_predicted)))

            confusion_mat = Utils.calculate_confusion_matrix(self.__y_test, y_test_predicted)
            print("SVM confusion Matrix")
            print(confusion_mat[0])
            print(confusion_mat[1])

            # print("SK Learn Metrics")
            # print(confusion_matrix(self.y_test, y_test_predicted))
            # print(classification_report(self.y_test, y_test_predicted))
        else:
            print("Train SVM before Testing.")

    def predict_sample(self, sample_data, return_lbl=False):
        try:
            if self.__trained_weights is not None:
                # min_max_scalar = Utils.calculate_min_max_scalar(pd.read_csv(self.training_data_dir))

                sample_data_copy = sample_data.copy(deep=True)

                sample_data_copy = Utils.normalize_dataset(sample_data_copy, self.__min_max_scalar)

                sample_data_copy.insert(loc=len(sample_data_copy.columns), column='intercept', value=1)

                y_predict = np.sign(np.dot(sample_data_copy, self.__trained_weights))

                if return_lbl is True:
                    if y_predict > 0:
                        return 'ASD'
                    else:
                        return 'Normal'

                return y_predict
            else:
                print("Train the SVM first OR load the trained weights")

        except Exception as e:
            print("Error occurred predicting the sample in SVM.")
            print('Exception', e)
            traceback.print_exc()

    def load_trained_weights(self, weights_fl=Weights_File):
        try:
            print("loading SVM trained weights")  # debug
            self.__trained_weights = np.loadtxt(weights_fl)
        except Exception as e:
            print("Error Occurred loading SVM weights", e)

    def __save_trained_weights(self, weights_fl):
        if self.__trained_weights is not None:
            w_fl = open(weights_fl, 'w')
            np.savetxt(w_fl, self.__trained_weights)
            w_fl.close()
            # self.__trained_weights = []
        else:
            print("Train SVM first.")


def stochastic_gradient_descent(features, labels):
    global Learning_rate
    max_epochs = 17000
    weights = np.zeros(features.shape[1])
    prev_cost = float("inf")
    cost_threshold = 0.0001
    count = 0

    for epoch in range(1, max_epochs):
        X, Y = shuffle_data(features=features, labels=labels)
        for indx, x in enumerate(X):
            ascent = calculate_cost_gradient(w=weights, x=x, y=Y[indx])
            weights = weights - (Learning_rate * ascent)

        if epoch == 2 ** count or epoch == max_epochs - 1:
            cost = calculate_hinge_loss(vector_w=weights, x=features, y=labels)
            print("Epoch No.: {} || Cost: {}".format(epoch, cost))

            # stopping condition
            if abs(prev_cost - cost) < (cost_threshold * prev_cost):
                return weights
            prev_cost = cost
            count += 1

    return weights


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


# W = 0 : max(0, 1 - Yi * (W * Xi)) = 0
# W = w - C * Yi * Xi
def calculate_cost_gradient(w, x, y):
    dist = 1 - (y * np.dot(w, x))
    dist_w = np.zeros(len(w))

    if max(0, dist) == 0:
        dist_w += w
    else:
        dist_w += w - (Reg_strength * y * x)

    return dist_w


#  shuffle features and labels in order
def shuffle_data(features, labels):
    index = np.random.permutation(len(labels))
    x = []
    y = []

    for i in index:
        x.append(features[i])
        y.append(labels[i])
    return x, y


def process_training_data(fl_dir, min_max_scalar):
    dataframe = pd.read_csv(fl_dir)

    #  calculates the min max value of each col in the dataframe
    # min_max_scalar = Utils.calculate_min_max_scalar(dataset=dataframe)

    # replacing the labels
    dataframe['feature_class']. \
        replace({'ASD': 1.0, 'TD': -1.0}, inplace=True)

    # splitting data into train and test
    x_train, x_test, y_train, y_test = Utils.train_test_split(dataframe=dataframe, test_size=0.2)

    # normalizing train and test data
    x_train = Utils.normalize_dataset(df=x_train, min_max=min_max_scalar)
    x_test = Utils.normalize_dataset(df=x_test, min_max=min_max_scalar)

    # insert intercept col 'b' in (W * Xi + b)
    x_train.insert(loc=len(x_train.columns), column='intercept', value=1)
    x_test.insert(loc=len(x_test.columns), column='intercept', value=1)

    return x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()


if __name__ == "__main__":
    svm_classifier = SupportVectorMachine("D:/TrainingDataset_YEAR_PROJECT/TrainingSet_Saliency_2.csv")
    svm_classifier.train_support_vector_machine()
    #
    # svm_classifier.save_trained_weights("SVM_Weights")
    # svm_classifier.load_trained_weights(
    #     "Trained_Classifiers/SVM_Weights")

    # svm_classifier.test_support_vector_machine()

    # ASD data
    # data = [[12, 2564, 213.666666666666, 29597, 2690.63636363636, 176.05935069273, 132.591777456996]]
    # df = pd.DataFrame(data, columns=["fixpoint_count", "total_duration", "mean_duration", "total_scanpath_len",
    #                                  "mean_scanpath_len", "mean_dist_centre", "mean_dist_mean_coord"])

    # result = svm_classifier.predict_sample(df)

    # train_SVM("D:/TrainingDataset_YEAR_PROJECT/TrainingSet.csv")
