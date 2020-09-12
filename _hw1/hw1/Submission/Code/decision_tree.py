from timeit import default_timer as timer
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

    try:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
    except Exception:
        precision = 1
        recall = 0

    # Calculating f1 score and returning it
    return 2 * ((precision * recall) \
                / (precision + recall))


def get_y_train_of_group(group):
    y_train_list = list()
    for each in group:
        y_train_list.append(each[len(each)-1])
    return y_train_list

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
    _gini = 0.0

    left_split = groups.get('left')
    right_split = groups.get('right')
    number_of_samples = len(left_split) + len(right_split)

    def gini_index_helper(group):
        # Calculating sum of the gini index for all groups
        nonlocal _gini
        length_of_current_group = len(group)
        if length_of_current_group != 0:
            score = 0 # will store the score for class
            for _class in classes:
                value = get_y_train_of_group(group).count(_class) / length_of_current_group
                score += value ** 2
            _gini += (length_of_current_group / number_of_samples) \
                     * (1.0 - score)

    gini_index_helper(left_split)
    gini_index_helper(right_split)
    return _gini


def get_split(formatted_data):
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
        'gini': None,
        'split_index': None,
        'split_value': None,
        'best_groups': None
    }

    classes = [0, 1]
    best_gini = sys.maxsize
    current_index = 0
    length_of_a_row = len(formatted_data[0]) - 1
    while current_index < length_of_a_row:
        for i, row in enumerate(formatted_data):
            current_value = row[current_index]
            groups = {
                'left': [],
                'right': []
            }
            for each in formatted_data:
                if each[current_index] >= current_value:
                    groups['right'].append(each)
                else:
                    groups['left'].append(each)
            current_gini = gini_index(groups, classes)
            if current_gini < best_gini:
                best_gini = current_gini
                result['gini'] = best_gini
                result['split_index'] = current_index
                result['split_value'] = current_value
                result['best_groups'] = groups
        current_index += 1

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
        self.min_size = min_size

    def split(self, current_node, current_depth):
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
        groups = current_node.pop('best_groups')
        left_node = groups.get('left')
        right_node = groups.get('right')
        if not (left_node or right_node):
            y_train_values = get_y_train_of_group(left_node + right_node)
            self.most_repeated_in_y_train_group = max(set(y_train_values), key=y_train_values.count)
            current_node['left'] = self.most_repeated_in_y_train_group
            current_node['right'] = self.most_repeated_in_y_train_group
            return
        if current_depth < self.max_depth:
            if len(left_node) > self.min_size:
                current_node['left'] = get_split(left_node)
                current_left_node = current_node['left']
                current_depth += 1
                self.split(current_left_node, current_depth)
            else:
                y_train_values = get_y_train_of_group(left_node)
                self.most_repeated_in_y_train_group = max(set(y_train_values), key=y_train_values.count)
                current_node['left'] = self.most_repeated_in_y_train_group

            if len(right_node) > self.min_size:
                current_node['right'] = get_split(right_node)
                current_right_node = current_node['right']
                current_depth += 1
                self.split(current_right_node, current_depth)
            else:
                y_train_values = get_y_train_of_group(right_node)
                self.most_repeated_in_y_train_group = max(set(y_train_values), key=y_train_values.count)
                current_node['right'] = self.most_repeated_in_y_train_group
        else:
            y_train_values = get_y_train_of_group(left_node)
            self.most_repeated_in_y_train_group = max(set(y_train_values), key=y_train_values.count)
            current_node['left'] = self.most_repeated_in_y_train_group
            y_train_values = get_y_train_of_group(right_node)
            self.most_repeated_in_y_train_group = max(set(y_train_values), key=y_train_values.count)
            current_node['right'] = self.most_repeated_in_y_train_group

        
    def fit(self, x_train, y_train):
        """
        Fitting the KNN classifier
        
        Hint: Build the decision tree using 
                splits recursively until a leaf
                node is reached

        """
        self.formatted_data = get_formatted_data(x_train, y_train)
        self.tree = get_split(self.formatted_data)
        self.groups = self.tree.get('best_groups')
        self.split(self.tree, 1)

    def predict(self, x_test):
        """
        Predicting the test data

        Hint: Run the test data through the decision tree built
                during training (self.tree)
        """
        index = self.tree.get('split_index')
        value = self.tree.get('split_value')

        if x_test[index] >= value:
            return self.tree.get('right')
        else:
            return self.tree.get('left')
      
                
def get_formatted_data(X, y):
    return np.insert(X, len(X[0]), y, axis=1)
    # data_hashmap = dict()
    # for index, (each_in_X, each_in_y) in enumerate(zip(X, y)):
    #     data_hashmap[index] = (each_in_X, each_in_y)
    # return data_hashmap

def main(X, y):
    # Example running the class DecisionTree
    print('==========-Decision Tree-==========')
    depths = [3, 6, 9, 12, 15]
    min_size = 10
    for depth in depths:
        print("------------------")
        print(f"Depth = {depth}")
        print("------------------")
        start_time = timer()

        dt = DecisionTree(max_depth=depth, min_size=min_size)
        kf = KFold(n_splits=5)
        kf.get_n_splits(X)
        total_f1_score = 0

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            dt.fit(X_train, y_train)
            predictions_for_each = []
            for each_X_test in X_test:
                predictions_for_each.append(dt.predict(each_X_test))
            total_f1_score += f1_score(y_test, predictions_for_each)

        print(f"Total Score = {total_f1_score/5}")
        print(f"Time Taken = {(timer() - start_time) * 1000} ms")
