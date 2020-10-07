from timeit import default_timer as timer
import csv
import re
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report, 
    confusion_matrix
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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

def load_data():
    def load(path):
        return pd.read_csv(path)

    def process_y(data):
        return np.array((data['Sentiment'] == 'Positive').values.astype(int))

    def process_x(data):
        return np.array(data["Review Text"])

    y_train = load("../../Data/Y_train.csv")
    y_train = process_y(y_train)
    y_valid = load("../../Data/Y_val.csv")
    y_valid = process_y(y_valid)

    X_train = load("../../Data/X_train.csv")
    X_train = process_x(X_train)
    X_valid = load("../../Data/X_val.csv")
    X_valid = process_x(X_valid)
    X_test = load("../../Data/X_test.csv")
    X_test = process_x(X_test)

    return X_train, y_train, X_valid, y_valid, X_test

def main():

    # FOR AVG VALIDATION SET

    print('------ Best Model File ------')

    

    X_train, y_train, X_valid, y_valid, X_test = load_data()

    bow = BagOfWords(vocabulary_size=10)
    X_train_representation = bow.transform(X_train)
    
    model = LogisticRegression()
    model.fit(X_train_representation, y_train)

    bow = BagOfWords(vocabulary_size=10)
    X_test_representation = bow.transform(X_test[:3299])

    predicted = model.predict(X_test_representation)
    np.savetxt("../Predictions/best.csv", predicted, delimiter=",")

    prob = model.predict_proba(X_test_representation)[:,0]

    precision = precision_score(y_valid, predicted)
    print(f"Precision = {precision}")
    recall = recall_score(y_valid, predicted)
    print(f"Recall = {recall}")
    f1 = f1_score(y_valid, predicted)
    print(f"F1 Score = {f1}")
    auc = roc_auc_score(y_valid, prob)
    print(f"AUC Score = {auc}")

    print("Confusion Matrix")
    print(confusion_matrix(y_valid, predicted))


if __name__ == '__main__':
    main()
