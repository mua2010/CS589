# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#Load data
X = np.load('../../Data/X_train.npy')
Y = np.load('../../Data/y_train.npy')

#%% Plotting mean of the whole dataset
mean = np.mean(X.T, axis=1)
reshaped = mean.reshape(28, 28)
plt.imshow(reshaped)
plt.title("Mean of whole Dataset Plot")
plt.savefig("../Figures/q3_mean_whole_dataset.png")

#%% Plotting each digit
plt.clf()
fig, ax = plt.subplots(5,2,figsize=(10,10))
fig.suptitle("Plot Each Digit")
each_digit_list = list()
counter = 0
while counter < 10:
    temp0 = X[np.where(Y==counter)]
    temp = np.mean(temp0.T, axis=1).reshape(28,28)
    each_digit_list.append(temp)
    counter += 1
counter = 0
for each in ax.flat:
    each.imshow(each_digit_list[counter])
    counter += 1
plt.savefig("../Figures/q3_each_digit.png")

#%% Center the data (subtract the mean)
centered_data = X - mean

#%% Calculate Covariate Matrix
cov_matrix = np.cov(centered_data.T)

#%% Calculate eigen values and vectors
values, vectors = np.linalg.eig(cov_matrix)

#%% Plot eigen values
plt.clf()
plt.plot(np.real(values))
plt.title("eigen values")
plt.savefig("../Figures/q3_eigen_values.png")

#%% Plot 5 first eigen vectors
plt.clf()
fig, ax = plt.subplots(5,1,figsize=(10,10))
fig.suptitle("5 first eigen vectors")
each_digit_list = list()
counter = 0
for each in ax.flat:
    placeholder = np.real(vectos.T[counter]).reshape(28,28)
    each.imshow(placeholder)
    counter += 1
plt.savefig("../Figures/q3_5_first_eigen_vectors.png")
breakpoint()

#%% Project to two first bases
two_first = vectors.T[:2]
projected = two_first.dot(centered_data.T)

#%% Plotting the projected data as scatter plot
plt.figure()