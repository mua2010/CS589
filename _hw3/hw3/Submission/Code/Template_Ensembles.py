# -*- coding: utf-8 -*-
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import math
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

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
    n_features = 300 # p
    N = 150 # Each part will be tried with 1 to 150 estimators
    number_of_trees_list = list(range(1, N + 1))
    
    # Train RF with m = sqrt(n_features) recording the errors (errors will be of size 150)

    test_classification_error_list_1 = list()
    m1 = int(math.sqrt(n_features)) # sqrt(p)
    for n in range(1, N + 1):
        clf = RandomForestClassifier(n_estimators=n, max_features=m1)
        clf.fit(train_X, train_y)
        y_pred = clf.predict(test_X)
        accuracy = accuracy_score(test_y, y_pred)
        test_classification_error_list_1.append(1 - accuracy)

    # Train RF with m = n_features recording the errors (errors will be of size 150)

    test_classification_error_list_2 = list()
    m2 = n_features # p
    for n in range(1, N + 1):
        clf = RandomForestClassifier(n_estimators=n, max_features=m2)
        clf.fit(train_X, train_y)
        y_pred = clf.predict(test_X)
        accuracy = accuracy_score(test_y, y_pred)
        test_classification_error_list_2.append(1 - accuracy)
    
    # Train RF with m = n_features/3 recording the errors (errors will be of size 150)

    test_classification_error_list_3 = list()
    m3 = int(n_features / 3) # p/3
    for n in range(1, N + 1):
        clf = RandomForestClassifier(n_estimators=n, max_features=m3)
        clf.fit(train_X, train_y)
        y_pred = clf.predict(test_X)
        accuracy = accuracy_score(test_y, y_pred)
        test_classification_error_list_3.append(1 - accuracy)
    
    # plot the Random Forest results

    plt.figure("Ensemble Random Forest Classifier")
    plt.plot(number_of_trees_list, test_classification_error_list_1, c='r', label=f"max_features = {m1} | sqrt(p)")
    plt.plot(number_of_trees_list, test_classification_error_list_2, c='g', label=f"max_features = {m2} | p")
    plt.plot(number_of_trees_list, test_classification_error_list_3, c='b', label=f"max_features = {m3} | p/3")
    plt.ylabel("Test Classification Error")
    plt.xlabel("# of Trees (n_estimators)")
    plt.legend(loc=1)
    plt.ylim(0,1)
    # plt.show()
    plt.savefig('../Figures/ensemble_randomforest_Q2.4_plot.png')
    
    # Train AdaBoost with max_depth = 1 recording the errors (errors will be of size 150)

    test_classification_error_list_1 = list()
    max_depth_1 = 1
    dtc = DecisionTreeClassifier(max_depth=max_depth_1)
    for n in range(1, N + 1):
        abc =AdaBoostClassifier(n_estimators=n, base_estimator=dtc, learning_rate=0.1)
        model = abc.fit(train_X, train_y)
        y_pred = model.predict(test_X)
        accuracy = accuracy_score(test_y, y_pred)
        test_classification_error_list_1.append(1 - accuracy)

    # Train AdaBoost with max_depth = 3 recording the errors (errors will be of size 150)

    test_classification_error_list_2 = list()
    max_depth_3 = 3
    dtc = DecisionTreeClassifier(max_depth=max_depth_3)
    for n in range(1, N + 1):
        abc =AdaBoostClassifier(n_estimators=n, base_estimator=dtc, learning_rate=0.1)
        model = abc.fit(train_X, train_y)
        y_pred = model.predict(test_X)
        accuracy = accuracy_score(test_y, y_pred)
        test_classification_error_list_2.append(1 - accuracy)

    # Train AdaBoost with max_depth = 5 recording the errors (errors will be of size 150)

    test_classification_error_list_3 = list()
    max_depth_5 = 5
    dtc = DecisionTreeClassifier(max_depth=max_depth_5)
    for n in range(1, N + 1):
        abc = AdaBoostClassifier(n_estimators=n, base_estimator=dtc, learning_rate=0.1)
        model = abc.fit(train_X, train_y)
        y_pred = model.predict(test_X)
        accuracy = accuracy_score(test_y, y_pred)
        test_classification_error_list_3.append(1 - accuracy)

    # plot the adaboost results

    plt.figure("Ensemble Ada Boost Classifier")
    plt.plot(number_of_trees_list, test_classification_error_list_1, c='r', label=f"max_depth = {max_depth_1}")
    plt.plot(number_of_trees_list, test_classification_error_list_2, c='g', label=f"max_depth = {max_depth_3}")
    plt.plot(number_of_trees_list, test_classification_error_list_3, c='b', label=f"max_depth = {max_depth_5}")
    plt.ylabel("Test Classification Error")
    plt.xlabel("# of Trees (n_estimators)")
    plt.legend(loc=1)
    plt.ylim(0,1)
    # plt.show()
    plt.savefig('../Figures/ensemble_adaboost_Q2.6_plot.png')

if __name__ == '__main__':
    main()