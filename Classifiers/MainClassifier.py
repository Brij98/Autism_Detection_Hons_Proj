from Classifiers.SupportVectorMachine import SupportVectorMachine
from Classifiers.RandomForest import RandomForest


class MainClassifier:
    def __init__(self):
        self.svm = SupportVectorMachine("D:/TrainingDataset_YEAR_PROJECT/TrainingSet.csv")

        self.random_forest = RandomForest(num_trees=10, min_samples_split=2, max_depth=8)
