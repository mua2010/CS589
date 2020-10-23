import pandas as pd
import numpy as np

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

import matplotlib.pyplot as plt

def load_data(dataset):
    """
    Load a pair of data X,y 

    Params
    ------
    dataset:    train/valid/test

    Return
    ------
    X:          shape (N, 240)
    y:          shape (N, 1)
    """
    X = pd.read_csv(f"Data/housing_data/{dataset}_x.csv", header=None).to_numpy()
    y = pd.read_csv(f"Data/housing_data/{dataset}_y.csv", header=None).to_numpy()

    return X,y

def score(model, X, y):
    """
    Score the model with X, y

    Params
    ------
    model:  the model to predict with
    X:      the data to score on
    y:      the true value y

    Return
    ------
    mae:    the mean absolute error
    """
    pass


def hyper_parameter_tuning(model_class, param_grid, train, valid):
    """
    Tune the hyper-parameter using training and validation data

    Params
    ------
    model_class:    the model class
    param_grid:     the hyper-parameter grid, dict
    train:          the training data (train_X, train_y)
    valid:          the validatation data (valid_X, valid_y)

    Return
    ------
    model:          model fit with best params
    best_param:     the best params
    """
    train_X, train_y = train
    valid_X, valid_y = valid

    # Set up the parameter grid
    param_grid = list(ParameterGrid(param_grid))

    # train the model with each parameter setting in the grid

    # choose the model with lowest MAE on validation set
    # then fit the model with the training and validation set (refit)

    # return the fitted model and the best parameter setting

def plot_mae_alpha(model_class, params, train, valid, test, title="Model"):
    """
    Plot the model MAE vs Alpha (regularization constant)

    Params
    ------
    model_class:    The model class to fit and plot
    params:         The best params found 
    train:          The training dataset
    valid:          The validation dataest
    test:           The testing dataset
    title:          The plot title

    Return
    ------
    None
    """
    train_X = np.concatenate([train[0], valid[0]], axis=0)
    train_y = np.concatenate([train[1], valid[1]], axis=0)

    # set up the list of alphas to train on

    # train the model with each alpha, log MAE

    # plot the MAE - Alpha
    

def main():
    """
    Load in data
    """
    train = load_data('train')
    valid = load_data('valid')
    test = load_data('test')

    """
    Define the parameter grid each each classifier
    e.g. lasso_grid = dict(alpha=[0.1, 0.2, 0.4],
                           max_iter=[1000, 2000, 5000])
    """
    
    ridge_grid = dict( 
        alpha = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
        max_iter=[1000, 2000, 3000, 4000, 5000, 7000, 10000]
    )
    lasso_grid = dict( 
        alpha = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
        max_iter=[1000, 2000, 3000, 4000, 5000, 7000, 10000]
    )
    ols_grid = dict(
        fit_intercept=[True, False]
    )



    # Tune the hyper-paramter by calling the hyper-parameter tuning function
    # e.g. lasso_model, lasso_param = hyper_parameter_tuning(Lasso, lasso_grid, train, valid)
    ols_model , ols_param = hyper_parameter_tuning(
        LinearRegression, 
        ols_grid, 
        train, valid
    )
    lasso_model, lasso_param = hyper_parameter_tuning(
        Lasso, 
        lasso_grid, 
        train, valid
    )
    ridge_model, ridge_param = hyper_parameter_tuning(
        Ridge, 
        ridge_grid, 
        train, valid
    )

    # Plot the MAE - Alpha plot by calling the plot_mae_alpha function
    # e.g. plot_mae_alpha(Lasso, lasso_param, train, valid, test, "Lasso")

    # plots of the MAE vs. regularization constant for the lasso regression model and
    # the ridge regression model
    plot_mae_alpha(Lasso, lasso_param, train, valid, test, "Lasso")
    plot_mae_alpha(Ridge, ridge_param, train, valid, test, "Ridge")

if __name__ == '__main__':
    main()
