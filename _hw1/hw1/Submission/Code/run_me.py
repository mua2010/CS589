import numpy as np
from knn import main as knn_main
from decision_tree import main as decision_tree_main
from best_model import main as best_model_main

def get_np_array_from_csv(path_to_csv):
	return np.genfromtxt(path_to_csv, delimiter=',')
X_t = get_np_array_from_csv('../../Data/x_train.csv')
y_t = get_np_array_from_csv('../../Data/y_train.csv')

# Best Model
best_model_main(X=X_t, y=y_t)

# KNN
knn_main(X=X_t, y=y_t)

# Decision Tree
decision_tree_main(X=X_t, y=y_t)