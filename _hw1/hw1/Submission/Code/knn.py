from timeit import default_timer as timer
from statistics import mode
# import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
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
        # Querying and returning indices of k nearest neighbors
        return self.tree.query(
            [x_instance], 
            k=self.K
        )[1]

    def fit(self, x_train, y_train):
        """
        Fitting the KNN classifier

        Hint:   Build a tree to get neighbors 
                faster at test time
        """
        self.tree = BallTree(x_train)
        self._y_train = y_train

    def predict(self, x_test):
        """
        Predicting the test data
        Hint:   Get the K-Neighbors, then generate
                predictions using the labels of the
                neighbors
        """
        y_pred = list()
        for x_instance in x_test:
            # getting the 1D array out of 2D
            neighbours = self.getKNeighbors(x_instance)[0]
            labels_of_neighbours = self._y_train[neighbours]
            most_frequent_label = mode(labels_of_neighbours)
            y_pred.append(most_frequent_label)
        return y_pred


def main(X, y):
    # Example running the class KNN
    print('==========-KNN-==========')
    neighbors_list = [3, 5, 10, 20, 25]

    for neighbors in neighbors_list:
        print("------------------")
        print(f"K = {neighbors} neighbors")
        print("------------------")
        start_time = timer()

        knn = KNN(n_neighbors=neighbors)
        kf = KFold(n_splits=5)
        kf.get_n_splits(X)
        total_f1_score = 0

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            total_f1_score += f1_score(y_test, y_pred)

        print(f"Total Score = {total_f1_score/5}")
        print(f"Time Taken = {timer() - start_time} seconds")

