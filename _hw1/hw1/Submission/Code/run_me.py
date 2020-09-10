import numpy as np
from knn import main as knn_main

X_t = np.genfromtxt('../../Data/x_train.csv', delimiter=',')
y_t = np.genfromtxt('../../Data/y_train.csv', delimiter=',')
knn_main(X=X_t, y=y_t)