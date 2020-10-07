# -*- coding: utf-8 -*-
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def load_data():
    """
    Helper function for loading in the data

    ------
    # of training samples: 63
    # of testing samples: 20
    ------
    """
    train_X = np.genfromtxt("../../Data/gene_data/gene_train_x.csv", delimiter= ",")
    train_y = np.genfromtxt("../../Data/gene_data/gene_train_y.csv", delimiter= ",")
    test_X = np.genfromtxt("../../Data/gene_data/gene_test_x.csv", delimiter= ",")
    test_y = np.genfromtxt("../../Data/gene_data/gene_test_y.csv", delimiter= ",")

    return train_X, train_y, test_X, test_y



def main():
    np.random.seed(0)
    train_X, train_y, test_X, test_y = load_data()
    
    N = 150 # Each part will be tried with 1 to 150 estimators
    
    # Train RF with m = sqrt(n_features) recording the errors (errors will be of size 150)
    
    # Train RF with m = n_features recording the errors (errors will be of size 150)
    
    # Train RF with m = n_features/10 recording the errors (errors will be of size 150)
    
    #plot the Random Forest results
    
    # Train AdaBoost with max_depth = 1 recording the errors (errors will be of size 150)

    # Train AdaBoost with max_depth = 3 recording the errors (errors will be of size 150)

    # Train AdaBoost with max_depth = 5 recording the errors (errors will be of size 150)

    # plot the adaboost results

if __name__ == '__main__':
    main()
