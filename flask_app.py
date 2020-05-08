import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB as NB
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
import math
import pickle
from flask import Flask
from flask import Flask, request, render_template
from ensemble_voting import EnsembleVoting

app = Flask(__name__)

pickle_off = open("EV.txt", "rb")
EV = pickle.load(pickle_off)


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST', 'GET'])
def predict():
    comment = request.form['input_text']
    rating = EV.ensemble_voting_predict(comment)

    return "Your prediction rating is " + str(rating)

@app.route("/clear", methods=['POST', 'GET'])
def clear():
    return render_template('index.html')

    
if __name__ == "__main__":
	app.run()
