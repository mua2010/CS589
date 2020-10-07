# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingClassifier
import pandas as pd
# feel free to import any sklearn model here


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

    # fit the model on the training data
    
    # Get and print f1-score on test data

if __name__ == '__main__':
    main()
