import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_path = "data.csv"
data_raw = pd.read_csv(data_path)
print("Number of rows in data =",data_raw.shape[0])
print("Number of columns in data =",data_raw.shape[1])
print("\n")
print("**Sample data:**")
data_raw.head()

categories = list(data_raw.columns.values)[3:12]
print(categories)
sns.set(font_scale = 2)
plt.figure(figsize=(20,30))
print(data_raw.iloc[:,3:12].sum())
ax= sns.barplot(categories, data_raw.iloc[:,3:12].sum().values)
plt.title("Review in each category", fontsize=24)
plt.ylabel('Number of Review', fontsize=18)
plt.xlabel('Experiance Type ', fontsize=18)
#adding the text labels
rects = ax.patches
labels = data_raw.iloc[:,3:].sum().values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=18)
#plt.show()


rowSums = data_raw.iloc[:,2:].sum(axis=1)
multiLabel_counts = rowSums.value_counts()
multiLabel_counts = multiLabel_counts.iloc[:]
sns.set(font_scale = 2)
plt.figure(figsize=(40,10))
ax = sns.barplot(multiLabel_counts.index, multiLabel_counts.values)
plt.title("Reviews having multiple labels ")
plt.ylabel('Number of reviews', fontsize=18)
plt.xlabel('Number of labels', fontsize=18)
#adding the text labels
rects = ax.patches
labels = multiLabel_counts.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
#plt.show()

#Data Pre-Processing

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
def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned
def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent
data['Review'] = data['Review'].str.lower()
data['Review'] = data['Review'].apply(cleanHtml)
data['Review'] = data['Review'].apply(cleanPunc)
data['Review'] = data['Review'].apply(keepAlpha)

#Removing stop words
stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)
data['Review'] = data['Review'].apply(removeStopWords)

#Stemming
stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence
data['Review'] = data['Review'].apply(stemming)

# test and train data partitioning...

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, random_state=42, test_size=0.10, shuffle=True)
print(train.shape)
print(test.shape)

train_text = train['Review']
test_text = test['Review']
#print("trian")
#print(train)
#print("test")
#print(test_text)

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
vectorizer.fit(train_text)
vectorizer.fit(test_text)
tfidf_vectorizer_file = open("tf_idf_vectorizer.pickle","wb")
pickle.dump(vectorizer, tfidf_vectorizer_file)
tfidf_vectorizer_file.close()
print(len(vectorizer.get_feature_names()))

x_train = vectorizer.transform(train_text)
#print("train_text")
#print(train_text)
#print("x_train")
#print(x_train)
y_train = train.drop(labels = ['Review'], axis=1)
print(y_train)
x_test = vectorizer.transform(test_text)
y_test = test.drop(labels = ['Review'], axis=1)

# Multiple Binary Classifications - (One Vs Rest Classifier)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from IPython.display import Markdown, display




#% time

# Using pipeline for applying logistic regression and one vs rest classifier
LogReg_pipeline = Pipeline([
    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
])

arrs = []
# print(str(pickle.dumps(LogReg_pipeline)))
# print(type(LogReg_pipeline))
pipeline_array = []
for category in categories:
    print('**Processing {} review...**'.format(category))

    # Training logistic regression model on train data
    # print("x_train")
    # print(x_train)
    LogReg_pipeline.fit(x_train, train[category])
    our_model = pickle.dumps(LogReg_pipeline)
    pipeline_array.append(our_model)
    # print(train[category])
    # calculating test accuracy
    # print("x_test")
    # print(x_test)
    # prediction = LogReg_pipeline.predict(x_test)
    # arrs.append(prediction)
    # print("Prediction: ")
    # print(prediction)
    # print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
    # print("\n")

pickle_out = open("trained_model.pickle", "wb")
pickle.dump(pipeline_array, pickle_out)
pickle_out.close()

print("Trained model saved in the file " + pickle_out.name)

