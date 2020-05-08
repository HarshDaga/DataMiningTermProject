import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB as NB
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
import math


class EnsembleVoting:
    # x_train, y_train, x_test, and y_test are all
    def __init__(self, x_train, y_train, x_test, y_test, use_percentage=1, num_of_classifier=3):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.num_of_classifier = num_of_classifier
        self.upper_index = math.ceil(x_train.shape[0] * use_percentage)
        self.increment = math.ceil(self.upper_index / num_of_classifier)
        self.x_df_container = []
        self.y_df_container = []
        self.vectorizers = self.instantiate_vectorizers()
        self.naive_bayes = self.instantiate_naive_bayes()

    def fit_all_naive_bayes(self):
        for i in range(self.num_of_classifier):
            train_vector = self.vectorizers[i].fit_transform(self.x_df_container[i]["comment"])
            target_train_vector = np.asarray(self.y_df_container[i]["rating"], dtype="|S6")
            self.naive_bayes[i].fit(train_vector, target_train_vector)

    def instantiate_naive_bayes(self):
        naive_bayes = []
        for i in range(self.num_of_classifier):
            t_naive_bayes = NB()
            naive_bayes.append(t_naive_bayes)

        return naive_bayes

    def instantiate_vectorizers(self):
        vectorizers = []
        for i in range(self.num_of_classifier):
            t_vectorizer = feature_extraction.text.CountVectorizer()
            vectorizers.append(t_vectorizer)

        return vectorizers

    def groom_data(self):
        split_indices = [0]
        current_index = 0
        while(True):
            current_index += self.increment
            if current_index <= self.upper_index:
                split_indices.append(current_index)
            else:
                split_indices.append(self.upper_index)
                break;

        for i in range(len(split_indices)-1):
            l = i
            r = i + 1
            t_x_train, t_y_train = self.x_train[split_indices[l]:split_indices[r], :], self.y_train[split_indices[l]:split_indices[r], :]
            t_x_train, t_y_train = pd.DataFrame(data=t_x_train, columns=["comment"]), pd.DataFrame(data=t_y_train, columns=["rating"])
            self.x_df_container.append(t_x_train)
            self.y_df_container.append(t_y_train)

    def ensemble_voting_predict(self, comment):
        input_np = np.array([[comment]])
        input_df = pd.DataFrame(data=input_np, columns=["comment"])
        ratings = []

        for i in range(self.num_of_classifier):
            t_input_vector = self.vectorizers[i].transform(input_df["comment"])
            t_pred = self.naive_bayes[i].predict(t_input_vector)
            rating = int(str(t_pred[0])[2])

            if int(str(t_pred[0])[2]) == 1 and str(t_pred[0])[3] == '0':
                rating = 10

            ratings.append(rating)

        return round(sum(ratings) / len(ratings))
