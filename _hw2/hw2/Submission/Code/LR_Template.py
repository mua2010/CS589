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
    # Compute scores
    scores = self.sigmoid(self.W * X.T)
    # Compute the loss
    loss = (-1 / N) \
            * np.sum(
                y * np.log(scores) 
                + (1 - y)
                    * np.log(1 - scores)
            ) \
            + self.reg * np.sum(self.W**2)
    # Compute gradients
    # Calculate dLdW meaning the gradient of loss function according to W 
    # you can use chain rule here with calculating each step to make your job easier
    dLdW = (- X.T * scores / N) \
            + (X.T * y / N)
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
      self.W += (learning_rate * dLdW) \
                 - (2 * learning_rate * self.reg * self.W)
      # printing loss, you can also print accuracy here after few iterations to see how your model is doing
    #   print("Epoch : ", i, " loss : ", loss)
      
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
    # get the scores (probabilities) for input data X and calculate the labels (0 and 1 for each input data) and return them
    probs = self.sigmoid(self.W * X.T)
    y_pred = list()
    for each in probs:
        if each < 0.5:
            y_pred.append(0)
        else:
            y_pred.append(1)
    
    return probs, np.array(y_pred)

def load_data():
    def load(path):
        return pd.read_csv(path)

    def process_y(data):
        return (data['Sentiment'] == 'Positive').values.astype(int)

    def process_x(data):
        return data["Review Text"]

    y_train = load("../../Data/Y_train.csv")
    y_train = process_y(y_train)
    y_valid = load("../../Data/Y_val.csv")
    y_valid = process_y(y_valid)

    X_train = load("../../Data/X_train.csv")
    X_train = process_x(X_train)
    X_valid = load("../../Data/X_val.csv")
    X_valid = process_x(X_valid)
    X_test = load("../../Data/X_test.csv")
    X_test = np.array(process_x(X_test))

    return X_train, y_train, X_valid, y_valid, X_test

_vocabulary_size = None

def cv_preprocess(X_train, X_valid, X_test):
    cv = CountVectorizer()
    X_train = cv.fit_transform(X_train)
    global _vocabulary_size
    _vocabulary_size = len(cv.get_feature_names())
    X_valid = cv.transform(X_valid)
    X_test = cv.transform(X_test)
    return X_train, X_valid, X_test

def main_loop_helper(lr, X_valid, X_train):
    return lr.predict(X_valid)[0], lr.predict(X_train)[0]

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

        # ---- Looping Regularization Weights -----
    test_train_rw = {
        'test_rw': list(),
        'train_rw': list()
    }
    regularization_weights = [0, 0.05, 0.1, 0.2]
    auc_rw_list = list()
    for reg in regularization_weights:
        lr = LogisticRegression(input_size=_vocabulary_size , reg=reg)
        lr.gradDescent(X_train, y_train, learning_rate=1, num_epochs=1000)
        y_prob, y_prob_train = main_loop_helper(lr, X_valid, X_train)
        auc = getauc(y_valid, y_prob)
        test_train_rw.get('test_rw').append(auc)
        test_train_rw.get('train_rw').append(getauc(y_train, y_prob_train))
        auc_rw_list.append(auc)
    print(f"Best regularization weight AUC = {max(auc_rw_list)}")
    plt.figure("ROC_AUC | Regularization Weights")
    plt.plot(regularization_weights, auc_rw_list)
    plt.xlabel("regularization weights")
    plt.ylabel("AUC")
    plt.savefig('../Figures/LR_looping_regularization_weights.png')
    # ---- Plotting Regularization Weights -----
    plt.figure("Plot Test.Train | Regularization Weights")
    plt.plot(regularization_weights, test_train_rw.get('test_rw'), 'g', label="validation-set AUC")
    plt.plot(regularization_weights, test_train_rw.get('train_rw'), 'm', label="train-set AUC")
    plt.ylabel("AUC")
    plt.xlabel("regularization weights")
    plt.savefig('../Figures/LR_test_train_Regularization_Weights.png')

    
    # ---- Looping Learning Rates -----
    test_train_lr = {
        'test_lr': list(),
        'train_lr': list()
    }
    learning_rates = [10**-4, 10**-1, 1, 10]
    auc_lr_list = list()
    for learning_rate in learning_rates:
        lr = LogisticRegression(input_size=_vocabulary_size, reg=0)
        #X, y, learning_rate, num_epochs
        lr.gradDescent(X_train, y_train, learning_rate=learning_rate, num_epochs=1000)
        y_prob, y_prob_train = main_loop_helper(lr, X_valid, X_train)
        auc = getauc(y_valid, y_prob)
        test_train_lr.get('test_lr').append(auc)
        test_train_lr.get('train_lr').append(getauc(y_train,y_prob_train))
        auc_lr_list.append(auc)
    print(f"Best learning rate AUC = {max(auc_lr_list)}")
    plt.figure("ROC_AUC | Learning Rates")
    plt.plot(learning_rates, auc_lr_list)
    plt.xlabel("Learning Rate")
    plt.ylabel("AUC")
    plt.savefig('../Figures/LR_looping_learning_rates.png')
    # ---- Plotting Learning Rates -----
    plt.figure("Plot Test.Train | Learning Rates")
    plt.plot(learning_rates, test_train_lr.get('test_lr'), 'g', label="validation-set AUC")
    plt.plot(learning_rates, test_train_lr.get('train_lr'), 'm', label="train-set AUC")
    plt.ylabel("AUC")
    plt.xlabel("Learning Rates")
    plt.savefig('../Figures/LR_test_train_Learning_Rates.png')

    
    # ---- Looping Num of Iterations -----
    test_train_noi = {
        'test_noi': list(),
        'train_noi': list()
    }
    num_of_iterations = [10, 750, 1000, 1500]
    auc_noi_list = list()
    for num_epochs in num_of_iterations:
        lr = LogisticRegression(input_size=_vocabulary_size, reg=0)
        lr.gradDescent(X_train, y_train, learning_rate=1, num_epochs=num_epochs)
        y_prob, y_prob_train = main_loop_helper(lr, X_valid, X_train)
        auc = getauc(y_valid, y_prob)
        test_train_noi.get('test_noi').append(auc)
        test_train_noi.get('train_noi').append(getauc(y_train, y_prob_train))
        auc_noi_list.append(auc)
    print(f"Best # of Iterations AUC = {max(auc_noi_list)}")
    plt.figure("ROC_AUC | # of Iterations")
    plt.plot(num_of_iterations, auc_noi_list)
    plt.xlabel("# of Iterations")
    plt.ylabel("AUC")
    plt.savefig('../Figures/LR_looping_num_of_Iterations.png')
    # ---- Plotting Num of Iterations -----
    plt.figure("Plot Test.Train | # of Iterations")
    plt.plot(num_of_iterations, test_train_noi.get('test_noi'), 'g', label="validation-set AUC")
    plt.plot(num_of_iterations, test_train_noi.get('train_noi'), 'm', label="train-set AUC")
    plt.ylabel("AUC")
    plt.xlabel("# of Iterations")
    plt.savefig('../Figures/LR_test_train_num_of_Iterations.png')

    # For part 1 & 3
    cm_lr = LogisticRegression(input_size=_vocabulary_size, reg=0)
    cm_lr.gradDescent(X_train, y_train, learning_rate=1, num_epochs=1000)
    y_pred_test = cm_lr.predict(X_valid)[1]
    print(f"{getauc(y_valid, y_prob)}")
    print("Confusion Matrix on the validation data with the best hyper parameters")
    print("learning_rate=1, num_epochs=1000, reg=0")
    print(conf_mat(y_valid, y_pred_test))


if __name__ == '__main__':
    main()



