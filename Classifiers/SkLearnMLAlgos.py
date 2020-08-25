import pandas as pd
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectFromModel
from sklearn.linear_model import LogisticRegression
import joblib
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from Classifiers import SupportVectorMachine

from sklearn.preprocessing import MinMaxScaler


def train_svm_model(fl_dir):
    #  X_train, X_test, Y_train, Y_test = SupportVectorMachine.proc_CSV_data(fl_dir)
    dataframe = pd.read_csv(fl_dir)
    dataframe["feature_class"]. \
        replace({"ASD": 0, "TD": 1},
                inplace=True)
    dataframe = dataframe.sample(frac=1)
    X = dataframe.drop(labels="feature_class", axis=1)
    Y = dataframe['feature_class']

    #  feature selection BEGIN
    # print("Shape before feature selection", X.shape)

    # L1-based feature selection
    # lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, Y)
    # model = SelectFromModel(lsvc, prefit=True)
    # X = model.transform

    # Univariate feature selection
    #  X = SelectKBest(chi2, k=6).fit_transform(X, Y)

    # Tree-based feature selection
    # clf = ExtraTreesClassifier(n_estimators=50)
    # clf = clf.fit(X, Y)
    # clf.feature_importances_
    # model = SelectFromModel(clf, prefit=True)
    # X = model.transform(X)

    # print("Shape after feature seleciton", X.shape)
    #  feature selection END

    #  train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

    #  SVM

    # scalar = MinMaxScaler()
    # X_train = pd.DataFrame(scalar.fit_transform(X_train.values))
    # X_test = pd.DataFrame(scalar.transform(X_test.values))

    # ns_probs = [0 for _ in range(len(Y_test))]
    # svm_model_linear = SVC(kernel='linear', C=1.0, probability=True).fit(X_train, Y_train)  # polynomial kernel
    #
    # # load the saved model
    # # load_model = joblib.load(filename=saved_mdl_path)
    #
    # svm_prediction = svm_model_linear.predict(X_test)
    #
    # accuracy = svm_model_linear.score(X_test, Y_test)
    # print(accuracy)  # debug
    #
    # # creating a confusion matrix
    # cm = confusion_matrix(Y_test, svm_prediction)
    #
    # print(cm)
    # print(classification_report(Y_test, svm_prediction))
    #
    # # predict probabilities
    # lr_probs = svm_model_linear.predict_proba(X_test)
    # # keep probabilities for the positive outcome only
    # lr_probs = lr_probs[:, 1]
    # # calculate scores
    # ns_auc = roc_auc_score(Y_test, ns_probs)
    # lr_auc = roc_auc_score(Y_test, lr_probs)
    # # summarize scores
    # print('No Skill: ROC AUC=%.3f' % (ns_auc))
    # print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # # calculate roc curves
    # ns_fpr, ns_tpr, _ = roc_curve(Y_test, ns_probs)
    # lr_fpr, lr_tpr, _ = roc_curve(Y_test, lr_probs)
    # # plot the roc curve for the model
    # pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    # pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # # axis labels
    # pyplot.xlabel('False Positive Rate')
    # pyplot.ylabel('True Positive Rate')
    #
    # # save the model
    # # saved_mdl_path = 'normal_leaf_model.sav'
    # # joblib.dump(svm_model_linear, saved_mdl_path)
    #
    # # show the legend
    # pyplot.legend()
    # # show the plot
    # pyplot.show()

    # RANDOM FOREST CLASSIFIER
    scalar = MinMaxScaler()
    X_train = pd.DataFrame(scalar.fit_transform(X_train.values))
    X_test = pd.DataFrame(scalar.transform(X_test.values))

    ns_probs = [0 for _ in range(len(Y_test))]
    regressor = RandomForestClassifier(n_estimators=18, max_depth=10).fit(X_train, Y_train)
    y_pred = regressor.predict(X_test)

    print(confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))
    # print(Y_test)
    # print(y_pred)
    print(SupportVectorMachine.calc_accuracy_score(Y_test, y_pred))

    accuracy = regressor.score(X_test, Y_test)
    print(accuracy)  # debug

    print(accuracy_score(Y_test, y_pred.round(), normalize=False))

    # NN..............................................................
    # mlp = MLPClassifier(hidden_layer_sizes=(7, 7, 7)).fit(X_train, Y_train)
    #
    # predictions = mlp.predict(X_test)
    # print(confusion_matrix(Y_test, predictions))
    # print(classification_report(Y_test, predictions))
    # print(SupportVectorMachine.calc_accuracy_score(Y_test, predictions))

    # KNN
    # scalar = MinMaxScaler()
    # X_train = pd.DataFrame(scalar.fit_transform(X_train.values))
    # X_test = pd.DataFrame(scalar.transform(X_test.values))

    # model = KNeighborsClassifier(n_neighbors=11).fit(X_train, Y_train)
    # y_pred = model.predict(X_test)
    # #
    # accuracy = model.score(X_test, Y_test)
    # print(accuracy)  # debug
    #
    # print(confusion_matrix(Y_test, y_pred))
    # print(classification_report(Y_test, y_pred))
    # accuracy = model.score(X_test, Y_test)
    # print(accuracy)  # debug

    # Naive Bayes
    # scalar = MinMaxScaler()
    # X_train = pd.DataFrame(scalar.fit_transform(X_train.values))
    # X_test = pd.DataFrame(scalar.transform(X_test.values))
    # gnb = GaussianNB()
    # y_pred = gnb.fit(X_train, Y_train).predict(X_test)
    # print(SupportVectorMachine.calc_accuracy_score(Y_test, y_pred))
    #
    # print(confusion_matrix(Y_test, y_pred))
    # print(classification_report(Y_test, y_pred))
    # accuracy = gnb.score(X_test, Y_test)
    # print(accuracy)  # debug

    # Logistic Regression
    # scalar = MinMaxScaler()
    # X_train = pd.DataFrame(scalar.fit_transform(X_train.values))
    # X_test = pd.DataFrame(scalar.transform(X_test.values))
    # logreg = LogisticRegression()
    #
    # # fit the model with data
    # logreg.fit(X_train, Y_train)
    #
    # y_pred = logreg.predict(X_test)
    #
    # print(confusion_matrix(Y_test, y_pred))
    # print(classification_report(Y_test, y_pred))
    # accuracy = logreg.score(X_test, Y_test)
    # print(accuracy)  # debug


if __name__ == "__main__":
    train_svm_model("D:/TrainingDataset_YEAR_PROJECT/TrainingSet.csv")
