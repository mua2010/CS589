# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    StackingClassifier,
    RandomForestClassifier
) 
import pandas as pd
from sklearn.metrics import f1_score
# feel free to import any sklearn model here
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


def load_data():
    """
    Helper function for loading in the data

    ------
    # of training samples: 419
    # of testing samples: 150
    ------
    """
    df = pd.read_csv("../../Data/breast_cancer_data/data.csv")

    cols = df.columns
    X = df[cols[2:-1]].to_numpy()
    y = df[cols[1]].to_numpy()
    y = (y=='M').astype(np.int) * 2 - 1

    train_X = X[:-150]
    train_y = y[:-150]

    test_X = X[-150:]
    test_y = y[-150:]

    return train_X, train_y, test_X, test_y

def main():
    np.random.seed(0)
    train_X, train_y, test_X, test_y = load_data()
    
    # Stacking models:
    # Create your stacked model using StackingClassifier
    models = [
        ('rf', RandomForestClassifier()),
        ('svm', SVC()),
        ('bayes', GaussianNB()),
        ('knn', KNeighborsClassifier()),
        ('cart', DecisionTreeClassifier())
    ]
    # define meta learner model
    final_estimator_model = LogisticRegression()
    # define the stacking ensemble
    sc = StackingClassifier(
        estimators=models, 
        final_estimator=final_estimator_model, 
        cv=5
    )

    # fit the model on the training data
    sc.fit(train_X, train_y)
    # Get and print f1-score on test data
    y_pred = sc.predict(test_X)
    print(f"f1 score = {f1_score(y_pred, test_y , average = 'weighted')}")

if __name__ == '__main__':
    main()
