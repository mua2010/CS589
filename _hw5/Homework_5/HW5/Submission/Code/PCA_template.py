# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#Load data
X = np.load('Data/X_train.npy')
Y = np.load('Data/y_train.npy')
#%% Plotting mean of the whole dataset
mean = np.mean(X.T, axis=1)
#%% Plotting each digit

#%% Center the data (subtract the mean)
centered_data = X - mean
#%% Calculate Covariate Matrix
cov_matrix = np.cov(centered_data.T)
#%% Calculate eigen values and vectors
values, vectors = np.eig(cov_matrix)
#%% Plot eigen values

#%% Plot 5 first eigen vectors

#%% Project to two first bases
projected = vectors.T.dot(centered_data.T)
#%% Plotting the projected data as scatter plot
