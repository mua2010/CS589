import numpy as np
from knn import main as knn_main
from decision_tree import main as decision_tree_main
from best_model import main as best_model_main

def get_np_array_from_csv(path_to_csv):
	return np.genfromtxt(path_to_csv, delimiter=',')
    
x_test = get_np_array_from_csv('../../Data/x_test.csv')
X_t = get_np_array_from_csv('../../Data/x_train.csv')
y_t = get_np_array_from_csv('../../Data/y_train.csv')

# Best Model
best_model_main(X=X_t, y=y_t, x_test=x_test)

# KNN
knn_main(X=X_t, y=y_t)

# Decision Tree (Not working!!!)
decision_tree_main(X=X_t, y=y_t)