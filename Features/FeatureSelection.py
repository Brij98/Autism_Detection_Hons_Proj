import pandas as pd
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, f_regression
import numpy as np


class FeatureSelection:

    def univariate_selection(self, data_file_name, k_best=10):
       
        dataframe = pd.read_csv(data_file_name)
        Y = dataframe.iloc[:, -1].to_numpy()
        X = dataframe.drop(dataframe.columns[-1], axis=1).to_numpy()

        test = SelectKBest(score_func=f_classif)
        fit_val = test.fit(X, Y)

        set_printoptions(precision=5)
        print(fit_val.scores_)
        features = fit_val.transform(X)


if __name__ == "__main__":
    feat_selection = FeatureSelection()
    feat_selection.univariate_selection(
        "C:/Users/Brijesh Prajapati/Documents/Projects/Autism_Detection_Hons_Proj/Features/"
        "Extracted_Feature_Files/TrainingSet_All.csv", 12)
