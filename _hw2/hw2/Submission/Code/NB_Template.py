import re
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score,
    accuracy_score,
    confusion_matrix
)



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

    def get_vocabulary(self):
        return self.vocabulary

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

    def set_priors(self):
        self.priors = Counter(self.y_train)

    def get_priors_log(self):
        return {
            'positive_probablity': math.log(self.priors.get(1)),
            'negative_probablity': math.log(self.priors.get(0))
        }

    def set_X_train_p_n(self):
        positive_samples = self.X_train[np.where(self.y_train==1)[0]]
        negative_samples = self.X_train[np.where(self.y_train==0)[0]]
        self.positives = np.sum(positive_samples, axis=0)
        self.positives_sum = np.sum(self.positives)
        self.negatives = np.sum(negative_samples, axis=0)
        self.negatives_sum = np.sum(self.negatives)

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
        self.cp = dict()
        self.cn = dict()
        for i in range(self.vocabulary_size):
            self.cp[i] = (self.positives[i] + (self.beta - 1)) \
                                     / (self.positives_sum \
                                        + (2 * (self.beta - 1)))
            self.cn[i] = (self.negatives[i] + (self.beta - 1)) \
                                     / (self.negatives_sum \
                                        + (2 * (self.beta - 1)))

    def predict(self, X_test):
        """
        Predict the X_test with the fitted model
        """
        ZERO = 0
        ONE = 1
        result = {
            'prediction': list(),
            'probablity': list()
        }
        for each in X_test:
            positive_probablity = self.get_priors_log().get('positive_probablity')
            negative_probablity = self.get_priors_log().get('negative_probablity')
            index = 0
            while index < len(each):
                positive_probablity += math.log(math.pow(self.cp.get(index), each[index]))
                negative_probablity += math.log(math.pow(self.cn.get(index), each[index]))
                index += 1
            if positive_probablity <= negative_probablity:
                result['prediction'].append(ZERO)
            elif positive_probablity > negative_probablity:
                result['prediction'].append(ONE)
            probablity = negative_probablity \
                         / (positive_probablity + negative_probablity)
            result['probablity'].append(probablity)
        result['prediction'] = np.array(result['prediction'])
        result['probablity'] = np.array(result['probablity'])
        return result
        


def confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix of the
        predictions with true labels
    """
    true = pd.Series(y_true, name='True')
    predicted = pd.Series(y_pred, name='Predicted')
    return pd.crosstab(true, predicted)


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
    def parse_and_get_nparray(data):
        ZERO = 0
        ONE = 1
        result = list()
        data = data.values
        for each_row in data:
            if each_row[1] == 'Positive':
                result.append(ONE)
            elif each_row[1] == 'Negative':
                result.append(ZERO)
        return np.array(result)
    
    def parse_and_get_review_nparray(data):
        return np.array(data["Review Text"])

    X_train = parse_and_get_review_nparray(pd.read_csv("../../Data/X_train.csv"))
    y_train = parse_and_get_nparray(pd.read_csv("../../Data/Y_train.csv"))
    X_valid = parse_and_get_review_nparray(pd.read_csv("../../Data/X_val.csv"))
    y_valid = parse_and_get_nparray(pd.read_csv("../../Data/Y_val.csv"))
    X_test = parse_and_get_review_nparray(pd.read_csv("../../Data/X_test.csv"))
    if return_numpy:
        cv = CountVectorizer()
        X_train = cv.fit_transform(X_train).toarray()
        global _vocabulary_size
        _vocabulary_size = len(cv.get_feature_names())
        X_valid = cv.transform(X_valid).toarray()
        X_test = cv.transform(X_test).toarray()
        return X_train, y_train, X_valid, y_valid, X_test
    elif not return_numpy:
        return X_train, y_train, X_valid, y_valid, X_test

_vocabulary_size = None

def main():
    # Load in data
    X_train, y_train, X_valid, y_valid, X_test = load_data(return_numpy=False)
    # Fit the Bag of Words model for Q1.1
    print("| Bag of Words model |")
    bow = BagOfWords(vocabulary_size=10)
    bow.fit(X_train[:100])
    representation = bow.transform(X_train[100:200])
    print(bow.get_vocabulary())
    print(np.sort(np.sum(representation, axis=0))[::-1])

    # Load in data
    X_train, y_train, X_valid, y_valid, X_test = load_data(return_numpy=True)
    print("| Naive Bayes model |")
    betas = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    roc_auc_list = []
    for beta in betas:
        # Fit the Naive Bayes model for Q1.3
        print(f"--- Beta = {beta} ---")
        nb = NaiveBayes(beta=beta, n_classes=2, vocabulary_size=_vocabulary_size)
        nb.fit(X_train, y_train)
        result = nb.predict(X_valid)
        y_pred = result['prediction']
        y_prob = result['probablity']
        print("Confusion Matrix")
        print(confusion_matrix(y_valid, y_pred))
        roc_score = roc_auc_score(y_valid, y_prob)
        print(f"ROC AUC score = {roc_score}")
        roc_auc_list.append(roc_score)
        f1 = f1_score(y_valid, y_pred)
        print(f"f1 score = {f1}")
        accuracy = accuracy_score(y_valid, y_pred)
        print(f"accuracy = {accuracy*100} percent")
        recall = recall_score(y_valid, y_pred)
        print(f"Recall Score = {recall}")
        precision = precision_score(y_valid, y_pred)
        print(f"Precision Score = {precision}")
        
    pyplot.figure("Roc Auc Score | Beta Plot")
    pyplot.plot(roc_auc_list, betas)
    pyplot.ylabel("beta")
    pyplot.xlabel("roc auc score")
    pyplot.show()

if __name__ == '__main__':
    main()
