import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold


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
    FRAUD  = 1
    NORMAL = 0

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for current_y_true, current_y_pred in zip(y_true, y_pred):
        if current_y_true == NORMAL:
            if current_y_pred == FRAUD:
                false_positives += 1
            elif current_y_pred == NORMAL:
                true_negatives += 1
        elif current_y_true == FRAUD:
            if current_y_pred == FRAUD:
                true_positives += 1
            elif current_y_pred == NORMAL:
                false_negatives += 1

    if true_positives != 0:
        # Calculating precision and recall
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
    else:
        precision = 1
        recall = 0

    # Calculating f1 score and returning it
    return 2 * ((precision * recall) \
              / (precision + recall))

def gini_index(groups, classes):
    """
    Function for calculating the gini index
        -- The goodness of the split

    Params
    ------
    groups  : A list containing two groups of samples
                resulted by the split
    classes : The classes in the classification problem
                e.g. [0,1]

    Return
    ------
    gini    : the gini index of the split
    """
    gini_index = 0.0

    number_of_samples = 0
    for group in groups:
        number_of_samples += len(group)
    
    # Calculating sum of the gini index for each group
    for group in groups:
        if len(group) != 0:
            score = 0 # will store the score for class
            for _class in classes:
                each_sub_group_list = list()
                for each in group:
                    each_sub_group_list.append(each[-1])
                value = each_sub_group_list.count(_class) / len(group)
                score += value ** 2
            gini_index += (len(group) / number_of_samples) \
                          * (1.0 - score)
    return gini_index

def get_split(x_train, y_train):
    """
    Function to generate the split which results in
        the best gini_index

    Params
    ------
    x_train : the input data (n_samples, n_features)
    y_train : the input label (n_samples,)

    Return
    ------
    {gini_index, split_index, split_value}
    """
    pass

##########################################################
# Alternatively combine gini_index into get_split and 
# find the split point using an array instead of a for
# loop, would speed up run time 
##########################################################

class DecisionTree(object):
    """
    The Decision Tree classifier
    """
    def __init__(self, max_depth, min_size):
        """
        Params
        ------
        max_depth   : the maximum depth of the decision tree
        min_size    : the minimum observation points on a 
                        leaf/terminal node
        """
        self.max_depth = max_depth
        self.min_size = 5

    def split(self, data, depth):
        """
        Function called recursively to split
            the data in order to build a decision
            tree.

        Params
        ------
        data    : {left_node, right_node, split_index, split_value}
        depth   : the current depth of the node in the decision tree

        Return
        ------
        """

    def fit(self, x_train, y_train):
        """
        Fitting the KNN classifier
        
        Hint: Build the decision tree using 
                splits recursively until a leaf
                node is reached

        """
        pass

    def predict(self, x_test):
        """
        Predicting the test data

        Hint: Run the test data through the decision tree built
                during training (self.tree)
        """
        pass

def main():
    # Example running the class DecisionTree
    dt = DecisionTree(max_depth=5)
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    score = f1_score(y_test, y_pred)








