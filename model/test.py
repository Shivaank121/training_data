import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_path = "data.csv"
test_data_path = "test.csv"
data_raw = pd.read_csv(data_path)
test_data = pd.read_csv(test_data_path)
print("**Sample data:**")
test_data.head()

categories = list(data_raw.columns.values)[3:12]
print(categories)

# Data Pre-Processing

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import warnings

data = data_raw
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext


def cleanPunc(sentence):  # function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned


def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent


test_data['Review'] = test_data['Review'].str.lower()
test_data['Review'] = test_data['Review'].apply(cleanHtml)
test_data['Review'] = test_data['Review'].apply(cleanPunc)
test_data['Review'] = test_data['Review'].apply(keepAlpha)

# Removing stop words
stop_words = set(stopwords.words('english'))
stop_words.update(
    ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'may', 'also', 'across',
     'among', 'beside', 'however', 'yet', 'within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)


def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)


test_data['Review'] = test_data['Review'].apply(removeStopWords)

# Stemming
stemmer = SnowballStemmer("english")


def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence


test_data['Review'] = test_data['Review'].apply(stemming)

# test and train data partitioning...

from sklearn.model_selection import train_test_split

# train, test = train_test_split(test_data, random_state=42, test_size=1.0, shuffle=True)
original_test_data = test_data
test = test_data
# print(train.shape)
print(test.shape)

test_text = test['Review']
# print("trian")
# print(train)
print("test")
print(test_text)

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

pickle_in = open("tf_idf_vectorizer.pickle", "rb")
vectorizer = pickle.load(pickle_in)
print(len(vectorizer.get_feature_names()))

# x_train = vectorizer.transform(train_text)
# print("train_text")
# print(train_text)
# print("x_train")
# print(x_train)
# y_train = train.drop(labels = ['Review'], axis=1)
# print(y_train)
x_test = vectorizer.transform(test_text)
y_test = test.drop(labels=['Review'], axis=1)

# Multiple Binary Classifications - (One Vs Rest Classifier)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from IPython.display import Markdown, display

# % time

# Using pipeline for applying logistic regression and one vs rest classifier
LogReg_pipeline = Pipeline([
    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
])

arrs = []
# our_model = pickle.dumps(LogReg_pipeline)
# LogReg_pipeline = pickle.loads(our_model)
pickle_in = open("trained_model.pickle", "rb")
pipeline_array = pickle.load(pickle_in)
# print(type(pickle.loads(pipeline_array[0])))
# pipeline_array = trained_model.read()
# print(str(pickle.dumps(LogReg_pipeline)))
# print(type(LogReg_pipeline))
for index in range(0, len(categories)):
    print('**Processing {} review...**'.format(categories[index]))

    # Training logistic regression model on train data
    # print("x_train")
    # print(x_train)
    # LogReg_pipeline.fit(x_train, train[category])
    LogReg_pipeline = pickle.loads(pipeline_array[index])
    # print(train[category])
    # calculating test accuracy
    # print("x_test")
    # print(x_test)
    prediction = LogReg_pipeline.predict(x_test)
    arrs.append(prediction)
    print("Prediction: ")
    print(prediction)
    # print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
    print("\n")

output_array = []
output_array.append(
    ['Review', 'Adventure & Outdoors', 'Spiritual', 'Nature & Retreat', 'Isolated or Hippie', 'Heritage',
     'Travel & Learn', 'Social Tourism (Volunteer & Travel)', 'Nightlife & Events', 'Shopping'])
test_review = original_test_data["Review"].values
for index in range(0, len(test_review)):
    row = []
    row.append(test_review[index])
    for arr in arrs:
        row.append(arr[index])
    output_array.append(row)

with open('output.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(output_array)
    # print(output_array)

print("Result is saved to the output.csv file")
