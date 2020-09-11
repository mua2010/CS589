import numpy as np
from knn import main as knn_main

def get_np_array_from_csv(path_to_csv):
	return np.genfromtxt(path_to_csv, delimiter=',')

X_t = get_np_array_from_csv('../../Data/x_train.csv')
y_t = get_np_array_from_csv('../../Data/y_train.csv')
knn_main(X=X_t, y=y_t)