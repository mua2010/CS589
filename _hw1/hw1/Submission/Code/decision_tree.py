import time
import sys
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


def gini_index(groups, classes, sub_y_train):
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
    _gini_index = 0.0

    left_split = groups['left']
    right_split = groups['right']
    number_of_samples = len(left_split) + len(right_split)
    
    def gini_index_helper(group):
        # Calculating sum of the gini index for each group
        nonlocal _gini_index
        length_of_current_group = len(group)
        if length_of_current_group != 0:
            score = 0 # will store the score for class
            for _class in classes:
                value = sub_y_train.count(_class) / length_of_current_group
                score += value ** 2
            _gini_index += (length_of_current_group / number_of_samples) \
                           * (1.0 - score)

    gini_index_helper(left_split)
    gini_index_helper(right_split)
    return _gini_index


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
    result = {
        'gini_index': None,
        'split_index': None,
        'split_value': None,
    }

    classes = [0, 1]
    best_gini = sys.maxsize
    current_index = 0
    length_of_a_row = len(x_train[0])
    while current_index < length_of_a_row:
        for i, row in enumerate(x_train):
            current_value = row[current_index]
            groups = {
                'left': [],
                'right': []
            }
            for row in x_train:
                if row[current_index] >= current_value:
                    groups['right'].append(row)
                else:
                    groups['left'].append(row)
            current_gini = gini_index(groups, classes, y_train[i])
            if current_gini < best_gini:
                best_gini = current_gini
                result['gini_index'] = best_gini
                result['split_index'] = current_index
                result['split_value'] = row[current_index]
                # best_index = current_index
                # best_value = row[best_index]
                best_group = groups # Dictionary

    return result


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








