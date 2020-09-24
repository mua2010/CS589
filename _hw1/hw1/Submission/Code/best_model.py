from timeit import default_timer as timer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
import csv
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def main(X, y, x_test):
    depths = [3, 6, 9, 12, 15]
    min_samples_split = 5
    runtimes = list()

    # FOR AVG VALIDATION SET

    print('==========-Best Model File - AVG Val. Set-==========')
    for depth in depths:
        print("------------------")
        print(f"Depth = {depth}")
        print("------------------")
        start_time = timer()

        kf = KFold(n_splits=5)
        kf.get_n_splits(X)
        total_precision = 0
        total_recall = 0
        total_f1_score = 0
        total_auc = 0
        predictions_list = []

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            start_time_for_fitting = timer()
            clf = DecisionTreeClassifier(max_depth=depth, min_samples_split=min_samples_split)
            clf.fit(X_train, y_train)
            end_time_for_fitting = start_time_for_fitting - timer()

            current_prediction = clf.predict(X_test)
            predictions_list.append(current_prediction)

            total_precision += precision_score(y_test, current_prediction)
            total_recall += recall_score(y_test, current_prediction)
            total_f1_score += f1_score(y_test, current_prediction)
            total_auc += roc_auc_score(y_test, current_prediction)

        print(f"Total Precision = {total_precision/5}")
        print(f"Total Recall = {total_recall/5}")
        print(f"Total f1 Score = {total_f1_score/5}")
        print(f"Total auc Score = {total_auc/5}")
        current_runtime = (timer() - start_time) * 1000
        runtimes.append(current_runtime)
        print(f"Time Taken = {current_runtime} ms")

    plt.figure("AVG Val. Set")
    plt.plot(depths, runtimes)
    plt.xlabel("Depth")
    plt.ylabel("Time (ms)")
    plt.show()

    # FOR FULL TRAINING SET

    print('==========-Best Model File - FULL Training Set-==========')
    for depth in depths:
        print("------------------")
        print(f"Depth = {depth}")
        print("------------------")

        total_precision = 0
        total_recall = 0
        total_f1_score = 0
        total_auc = 0

        clf = DecisionTreeClassifier(max_depth=depth, min_samples_split=min_samples_split)
        clf.fit(X, y)
        current_prediction = clf.predict(X)

        total_precision += precision_score(y, current_prediction)
        total_recall += recall_score(y, current_prediction)
        total_f1_score += f1_score(y, current_prediction)
        total_auc += roc_auc_score(y, current_prediction)

        print(f"Total Precision = {total_precision/5}")
        print(f"Total Recall = {total_recall/5}")
        print(f"Total f1 Score = {total_f1_score/5}")
        print(f"Total auc Score = {total_auc/5}")

    # Saving the prediction with best depth
    clf = DecisionTreeClassifier(max_depth=6, min_samples_split=5)
    clf.fit(X, y)
    np.savetxt("../Predictions/best.csv", clf.predict(x_test), delimiter=",")
