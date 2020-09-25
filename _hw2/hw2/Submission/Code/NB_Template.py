import re
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score



class BagOfWords(object):
    """
    Class for implementing Bag of Words
     for Q1.1
    """
    def __init__(self, vocabulary_size):
        """
        Initialize the BagOfWords model
        """
        self.vocabulary = list()
        self.vocabulary_size = vocabulary_size

    def preprocess(self, text):
        """
        Preprocessing of one Review Text
            - convert to lowercase
            - remove punctuation
            - empty spaces
            - remove 1-letter words
            - split the sentence into words

        Return the split words
        """
        
        valid_text_list = re.sub(r'[^\w\s]', '', text.lower()).split()
        return [word for word in valid_text_list if len(word) > 1]

    def fit(self, X_train):
        """
        Building the vocabulary using X_train
        """
        vocabulary_list = list()
        for review in X_train:
            preprocessed_review = self.preprocess(review)
            vocabulary_list.extend(preprocessed_review)
        
        sorted_vocabulary_list = Counter(vocabulary_list).most_common()

        for count in range(self.vocabulary_size):
            self.vocabulary.append(sorted_vocabulary_list[count][0])

        
        
    def transform(self, X):
        """
        Transform the texts into word count vectors (representation matrix)
            using the fitted vocabulary
        """
        representation_matrix = list()
        for each_review in X:
            preprocessed_review = self.preprocess(each_review)
            vector = np.zeros(self.vocabulary_size)
            for word in preprocessed_review:
                try:
                    vector[self.vocabulary.index(word)] += 1
                except:
                    continue
            representation_matrix.append(vector)
        return np.array(representation_matrix)

class NaiveBayes(object):
    def __init__(self, beta=1, n_classes=2, vocabulary_size=None):
        """
        Initialize the Naive Bayes model
            w/ beta and n_classes
        """
        self.beta = beta
        self.n_classes = n_classes
        self.vocabulary_size = vocabulary_size
        self.conditionals_positives = dict()
        self.conditionals_negatives = dict()

    def set_priors(self):
        self.priors = Counter(self.y_train)

    def set_X_train_p_n(self):
        positive_samples = self.X_train[np.where(self.y_train==1)[0]]
        negative_samples = self.X_train[np.where(self.y_train==0)[0]]
        self.positives = np.sum(positive_samples, axis=0)
        self.negatives = np.sum(negative_samples, axis=0)

    def fit(self, X_train, y_train):
        """
        Fit the model to X_train, y_train
            - build the conditional probabilities
            - and the prior probabilities
        """
        self.y_train = y_train
        self.X_train = X_train
        self.set_priors()
        number_of_unique_labels =  len(self.priors.keys())
        number_of_labels = len(y_train)
        for label in self.priors.keys():
            self.priors[label] += (self.beta - 1) \
                                 / (number_of_labels 
                                    + ((self.beta - 1) \
                                    * number_of_unique_labels))
        self.set_X_train_p_n()
        


    def predict(self, X_test):
        """
        Predict the X_test with the fitted model
        """
        pass


def confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix of the
        predictions with true labels
    """
    pass


def load_data(return_numpy=False):
    """
    Load data

    Params
    ------
    return_numpy:   when true return the representation of Review Text
                    using the CountVectorizer or BagOfWords
                    when false return the Review Text

    Return
    ------
    X_train
    y_train
    X_valid
    y_valid
    X_test
    """
    pass



def main():
    # Load in data
    X_train, y_train, X_valid, y_valid, X_test = load_data(return_numpy=False)
        
    # Fit the Bag of Words model for Q1.1
    bow = BagOfWords(vocabulary_size=10)
    bow.fit(X_train[:100])
    representation = bow.transform(X_train[100:200])

    # Load in data
    X_train, y_train, X_valid, y_valid, X_test = load_data(return_numpy=True)

    betas = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]#[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    ROC_AUC_list = []
    for beta in betas:
        # Fit the Naive Bayes model for Q1.3
        nb = NaiveBayes(beta=beta, n_classes=2, vocabulary=vocabulary)
        nb.fit(X_train, y_train)
        y_pred = nb.predict(X_valid)
        # print(confusion_matrix(y_valid, y_pred))


if __name__ == '__main__':
    main()
