import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from Classifiers import Utils

Learning_rate = 0.000001
Reg_strength = 10000  # C


#   SVM
#   1/2 * ||W||^2 + C(1/N * sum(max(0, 1 - Yi * (W * Xi +b))))

class SupportVectorMachine:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.trained_weights = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def train_support_vector_machine(self):
        print("Training SupportVectorMachine...")

        self.x_train, self.x_test, self.y_train, self.y_test = process_training_data(fl_dir=self.data_dir)
        self.trained_weights = stochastic_gradient_descent(self.x_train.to_numpy(), self.y_train.to_numpy())

        print(self.trained_weights)

        print("Finished Training SupportVectorMachine")

    def test_support_vector_machine(self):
        if self.trained_weights is not None and self.x_test is not None and self.y_test is not None:
            y_test_predicted = np.array([])

            # print(type(self.x_test))
            # print(type(self.y_test))

            self.x_train = self.x_train.to_numpy()
            self.y_test = self.y_test.to_numpy()

            # for i in range(self.x_test.shape[0]):
            for r in self.x_test.to_numpy():
                y_predict = np.sign(np.dot(r, self.trained_weights))
                y_test_predicted = np.append(y_test_predicted, y_predict)

            print("accuracy of the model: {}".format(Utils.calculate_accuracy_score(self.y_test, y_test_predicted)))

            confusion_mat = Utils.calculate_confusion_matrix(self.y_test, y_test_predicted)
            print(confusion_mat[0])
            print(confusion_mat[1])

            # print("SK Learn Metrics")
            # print(confusion_matrix(self.y_test, y_test_predicted))
            # print(classification_report(self.y_test, y_test_predicted))
        else:
            print("Train SVM before Testing.")

    def predict_sample(self, sample_data):
        try:
            min_max_scalar = Utils.calculate_min_max_scalar(pd.read_csv(self.data_dir))
            sample_data = Utils.normalize_dataset(sample_data, min_max_scalar)

            sample_data.insert(loc=len(sample_data.columns), column='intercept', value=1)

            y_predict = np.sign(np.dot(sample_data, self.trained_weights))
            if y_predict > 0:
                print('ASD')
            else:
                print('Normal')

            return y_predict

        except Exception as e:
            print("Error occurred predicting the sample. Error in predict_sample()")
            print('Exception', e)

    def load_trained_weights(self, weights_fl):
        pass

    def save_trained_weights(self, weights_fl):
        if self.trained_weights is not None:
            w_fl = open('SVM_Trained_Weight', 'w')
            np.savetxt(w_fl, self.trained_weights)
            w_fl.close()
        else:
            print("Train SVM first.")


def stochastic_gradient_descent(features, labels):
    global Learning_rate
    max_epochs = 17000
    weights = np.zeros(features.shape[1])
    prev_cost = float("inf")
    cost_threshold = 0.001
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


def process_training_data(fl_dir):
    dataframe = pd.read_csv(fl_dir)

    #  calculates the min max value of each col in the dataframe
    min_max_scalar = Utils.calculate_min_max_scalar(dataset=dataframe)

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

    return x_train, x_test, y_train, y_test


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

    test_scalar = Utils.calculate_min_max_scalar(dataframe.drop(labels='feature_class', axis=1))
    print("min_max", test_scalar)
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

    print(type(x_train))  # debug
    # print(type(y_train))  # debug

    # normalizing attempt
    x_train_1 = Utils.normalize_dataset(x_train, test_scalar)
    x_test_1 = Utils.normalize_dataset(x_test, test_scalar)

    print("X_train_1_head", x_train_1)
    print("X_test_1_head", x_test_1)

    # normalizing the  Data
    scalar = MinMaxScaler()
    x_train = pd.DataFrame(scalar.fit_transform(x_train.values))
    x_test = pd.DataFrame(scalar.transform(x_test.values))

    print("X_train_head", x_train.head())
    print("X_test_head", x_test.head())

    x_train.insert(loc=len(x_train.columns), column='intercept', value=1)
    x_test.insert(loc=len(x_test.columns), column='intercept', value=1)

    return x_train, x_test, y_train, y_test


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


if __name__ == "__main__":
    svm_classifier = SupportVectorMachine(data_dir="D:/TrainingDataset_YEAR_PROJECT/TrainingSet.csv")
    svm_classifier.train_support_vector_machine()
    svm_classifier.test_support_vector_machine()


    # ASD data
    data = [[12, 2564, 213.666666666666, 29597, 2690.63636363636, 176.05935069273, 132.591777456996]]
    df = pd.DataFrame(data, columns=["fixpoint_count", "total_duration", "mean_duration", "total_scanpath_len",
                          "mean_scanpath_len", "mean_dist_centre", "mean_dist_mean_coord"])

    result = svm_classifier.predict_sample(df)

    # train_SVM("D:/TrainingDataset_YEAR_PROJECT/TrainingSet.csv")
