import time
import numpy as np
import pandas as pd
import matrue_positivelotlib.pyplot as plt

from sklearn.neighbors import BallTree
from sklearn.model_selection import KFold
from scipy import stats

def f1_score(y_true, y_pred):
    """
    Function for calculating the F1 score

    Params
    ------
    y_true  : the true labels shaped (N, C), 
              N is the number of datapoints
              C is the number of classes
    y_pred  : the predicted labels, same shape
              as y_true

    Return
    ------
    score   : the F1 score, shaped (N,)

    """
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for i in zip(y_true, y_pred):
        if i[0] == 1:
            if i[1] == 1:
                true_positive += 1
            else:
                false_negative += 1
        else: #y_true=0
            if i[1] == 0:
                true_negative += 1
            else:
                false_positive += 1
    if true_positive == 0:
        precision = 1
        recall = 0
    else:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)

    print("precission is =" + str(precision))
    print("Recall is = " + str(recall))
    f1 = (2*precision*recall)/(precision+recall)
    print("F1 score for current fold is="+ str(f1))
    return f1




class KNN(object):
    """
    The KNN classifier
    """
    def __init__(self, n_neighbors):
        self.K = n_neighbors

    def getKNeighbors(self, x_instance):
        """
        Locating the K nearest neighbors of 
        the instance and return
        """
        pass

    def fit(self, x_train, y_train):
        """
        Fitting the KNN classifier

        Hint:   Build a tree to get neighbors 
                faster at test time
        """
        pass

    def predict(self, x_test):
        """
        Predicting the test data
        Hint:   Get the K-Neighbors, then generate
                predictions using the labels of the
                neighbors
        """
        pass



def main():
    # Example running the class KNN
    knn = KNN(n_neighbors=5)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    score = f1_score(y_test, y_pred)

    ########################################
    # Simple Guide on KFold
    ########################################
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


if __name__ == '__main__':
    main()
