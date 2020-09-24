from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score

def getauc(y_true, probs):
    """
    Use sklearn roc_auc_score to get the auc given predicted probabilities and the true labels
    
    Args:
        - y_true: The true labels for the data
        - probs: predicted probabilities
    """
    #TODO: return auc using sklearn roc_auc_score

def conf_mat(y_true, y_pred):
    """
    The method for calculating confusion matrix, you have to implement this by yourself.
    
    Args:
        - y_true: the true labels for the data
        - y_pred: the predicted labels
    """
    #TODO: compute and return confusion matrix
    
class LogisticRegression(object):
  def __init__(self, input_size, reg=0.0, std=1e-4):
    """
    Initializing the weight vector
    
    Input:
    - input_size: the number of features in dataset, for bag of words it would be number of unique words in your training data
    - reg: the l2 regularization weight
    - std: the variance of initialized weights
    """
    self.W = std * np.random.randn(input_size)
    self.reg = reg
    
  def sigmoid(self,x):
    """
    Compute sigmoid of x
    
    Input:
    - x: Input data of shape (N,)
    
    Returns:
    - sigmoid of each value in x with the same shape as x (N,)
    """
    #TODO: write sigmoid function

  def loss(self, X, y):
    """
    Compute the loss and gradients for your logistic regression.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: A numpy array f shape (N,) giving training labels.

    Returns:
    - loss: Loss (data loss and regularization loss) for the entire training samples
    - dLdW: gradient of loss with respect to W
    """
    N, D = X.shape
    reg = self.reg
    
    #TODO: Compute scores
    
    
    #TODO: Compute the loss

    #TODO: Compute gradients
    # Calculate dLdW meaning the gradient of loss function according to W 
    # you can use chain rule here with calculating each step to make your job easier
    
    
    return loss, dLdW

  def gradDescent(self,X, y, learning_rate, num_epochs):
    """
    We will use Gradient Descent for updating our weights, here we used gradient descent instead of stochastic gradient descent for easier implementation
    so you will use the whole training set for each update of weights instead of using batches of data.
    
    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - num_epochs: integer giving the number of epochs to train
    
    Returns:
    - loss_hist: numpy array of size (num_epochs,) giving the loss value over epochs
    """
    N, D = X.shape
    loss_hist = np.zeros(num_epochs)
    for i in range(num_epochs):
      #TODO: implement steps of gradient descent
      
      # printing loss, you can also print accuracy here after few iterations to see how your model is doing
      print("Epoch : ", i, " loss : ", loss)
      
    return loss_hist

  def predict(self, X):
    """
    Use the trained weights to predict labels for data given as X

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
        - probs: A numpy array of shape (N,) giving predicted probabilities for each of the elements of X.
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of the elements of X. You can get this by putting a threshold of 0.5 on probs
    """
    #TODO: get the scores (probabilities) for input data X and calculate the labels (0 and 1 for each input data) and return them
    
    
    return probs, y_pred

def main():
    # Load training data
    train_X = np.load('Data/X_train.npy')
    train_Y = np.load('Data/y_train.npy')
    
    #Binarize the training labels, Positive will be 1 and Negative will be 0
    y_train = (y_train['Sentiment'] == 'Positive').values.astype(int)

    # Load validation data
    X_val = pd.read_csv('Data/X_val.csv')
    y_val = pd.read_csv('Data/Y_val.csv')
    
    #Binarize the validation labels, Positive will be 1 and Negative will be 0
    y_val = (y_val['Sentiment'] == 'Positive').values.astype(int)
    
    #TODO: Preprocess the data, here we will only select Review Text column in both train and validation and use CountVectorizer from sklearn to get bag of word representation of the review texts
    # Careful that you should fit vectorizer only on train data and use the same vectorizer for transforming X_train and X_val 
    
    # Write a for loop for each hyper parameter here each time initialize logistic regression train it on the train data and get auc on validation data and get confusion matrix using the best hyper params 
    
if __name__ == '__main__':
    main()



