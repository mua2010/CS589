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
    #return auc using sklearn roc_auc_score
    return roc_auc_score(y_true, probs)

def conf_mat(y_true, y_pred):
    """
    The method for calculating confusion matrix, you have to implement this by yourself.
    
    Args:
        - y_true: the true labels for the data
        - y_pred: the predicted labels
    """
    # compute and return confusion matrix
    true = pd.Series(y_true, name='True')
    predicted = pd.Series(y_pred, name='Predicted')
    return pd.crosstab(true, predicted)
    
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
    # write sigmoid function
    return 1 / (1 + np.exp(-x))

    sig =  1/(1+np.exp(-x))
    for i, each in enumerate(sig):
      if each >= 0.999999:
        sig[i] = 0.999999
      if each <= 0.000001:
        sig[i] = 0.000001
    return np.array(sig)

  def loss(self, X, y):
    """
    Compute the loss and gradients for your logistic regression.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: A numpy array f shape (N,) giving training labels.

    Returns:
    - loss: Loss (data loss and regularization loss) for the entire training samples
    - dLdW: gradient of loss with respect to W

     lambda ||w||^2_2 -> np.sum(self.W**2)
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
      # implement steps of gradient descent
      loss, dLdW = self.loss(X, y)
      self.W -= (learning_rate * dLdW)
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

def load_data():
    def load(path):
        return pd.read_csv(path)

    def process_y(data):
        return np.array((data['Sentiment'] == 'Positive').values.astype(int))

    def process_x(data):
        return np.array(data["Review Text"])

    y_train = load("./Data/Y_train.csv")
    y_train = process_y(y_train)
    y_valid = load("./Data/Y_val.csv")
    y_valid = process_y(y_valid)

    X_train = load("./Data/X_train.csv")
    X_train = process_x(X_train)
    X_valid = load("./Data/X_val.csv")
    X_valid = process_x(X_valid)
    X_test = load("./Data/X_test.csv")
    X_test = process_x(X_test)

    return X_train, y_train, X_valid, y_valid, X_test

_vocabulary_size = None

def cv_preprocess(X_train, X_valid, X_test):
    cv = CountVectorizer()
    X_train = cv.fit_transform(X_train)
    global _vocabulary_size
    _vocabulary_size = len(cv.get_feature_names())
    X_valid = cv.transform(X_valid).toarray()
    X_test = cv.transform(X_test).toarray()
    return X_train.toarray(), X_valid, X_test

def main():
    # Load training data
    #Binarize the training labels, Positive will be 1 and Negative will be 0
    # Load validation data
    #Binarize the validation labels, Positive will be 1 and Negative will be 0
    # Preprocess the data, here we will only select Review Text column in both train and validation and use CountVectorizer from sklearn to get bag of word representation of the review texts
    # Careful that you should fit vectorizer only on train data and use the same vectorizer for transforming X_train and X_val 
    X_train, y_train, X_valid, y_valid, X_test = load_data()
    X_train, X_valid, X_test = cv_preprocess(X_train, X_valid, X_test)
    
    # Write a for loop for each hyper parameter here each time initialize logistic regression train it on the train data and get auc on validation data and get confusion matrix using the best hyper params 
    
if __name__ == '__main__':
    main()



