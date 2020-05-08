# The serialize properly, this file has to be used instead of Jupyter Notebook.
import numpy as np
import pandas as pd 
import string
import re
import unidecode
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB as NB
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
import math
from ipywidgets import widgets
from IPython.display import display
import pickle


review_df = pd.read_csv("boardgamegeek-reviews/bgg-13m-reviews.csv")



review_df.dropna(subset=["comment"], inplace=True)
review_df["comment"] = review_df["comment"].str.lower()
def functuation_removal(comment):
    return comment.translate(comment.maketrans('','', string.punctuation))

review_df["comment"] = review_df["comment"].apply(lambda comment:functuation_removal(comment))

def miscellaneous_removal(comment):
    pattern_html = re.compile(r'<.*?>')
    pattern_url = re.compile(r'https?://\S+\www\.\S+')
    comment = pattern_html.sub(r'', comment)
    comment = pattern_url.sub(r'', comment)
    comment = unidecode.unidecode(comment)
    return comment

review_df["comment"] = review_df["comment"].apply(lambda comment: miscellaneous_removal(comment))

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

def stop_words_removal(comment):
    return " ".join([word for word in tokenizer.tokenize(comment) if word not in stop_words])

review_df["comment"] = review_df["comment"].apply(stop_words_removal)

nltk.download()
words = set(nltk.corpus.words.words())

def remove_non_english_words(comment):
    return " ".join(w for w in nltk.wordpunct_tokenize(comment) if w.lower() in words or not w.isalpha())

review_df["comment"] = review_df["comment"].apply(remove_non_english_words)


review_df["rating"] = review_df["rating"].round(0).astype(int)

review_df.drop(review_df.columns[0], axis=1, inplace=True) # This removes the original index column

review_df.drop(review_df.columns[3], axis=1, inplace=True)

review_df.drop(review_df.columns[0], axis=1, inplace=True)

review_df.drop(review_df.columns[2], axis=1, inplace=True)

review_df.head()

review_df = review_df.sample(frac=1).reset_index(drop=True)

review_np = review_df.to_numpy()

x = review_np[:, 1:]
y = review_np[:, :1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
from ensemble_voting import EnsembleVoting

EV = EnsembleVoting(x_train, y_train, x_test, y_test, use_percentage=1)
EV.groom_data()
EV.fit_all_naive_bayes()
EV.clear_data_set()

from sklearn.externals import joblib
joblib.dump(EV, 'EV.pkl')

with open('EV.txt', 'wb') as fh:
    pickle.dump(EV, fh)