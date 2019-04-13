from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import csv
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
# import json

analyser = SentimentIntensityAnalyzer()


def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))


data_path = "data.csv"
data_raw = pd.read_csv(data_path)
# print("Number of rows in data =",data_raw.shape[0])
# print("Number of columns in data =",data_raw.shape[1])

categories = list(data_raw.columns.values)[3:12]
# print(categories)

#Data Pre-Processing

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


stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)

def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)


data['Review'] = data['Review'].apply(removeStopWords)


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


data['Review'] = data['Review'].apply(stemming)

train, test = train_test_split(data, random_state=42, test_size=0.30, shuffle=True)
original_test_data = test
# print(train.shape)
# print(test.shape)

train_text = train['Review']
test_text = test['Review']
# print("trian")
# print(train)
# print("test")
# print(test_text)

vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
vectorizer.fit(train_text)
vectorizer.fit(test_text)


x_train = vectorizer.transform(train_text)
# print("train_text")
# print(train_text)
#print("x_train")
# print(x_train)
y_train = train.drop(labels = ['Review'], axis=1)
#print(y_train)
x_test = vectorizer.transform(test_text)
y_test = test.drop(labels = ['Review'], axis=1)

# Using pipeline for applying logistic regression and one vs rest classifier
LogReg_pipeline = Pipeline([
    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)),
])

arrs = []
for category in categories:
    #print('**Processing {} review...**'.format(category))

    # Training logistic regression model on train data
    # print("x_train")
    # print(x_train)
    LogReg_pipeline.fit(x_train, train[category])

    # calculating test accuracy
    # print("x_test")
    # print(x_test)
    prediction = LogReg_pipeline.predict(x_test)
    arrs.append(prediction)
    #print("Prediction: ")
    #print(prediction)
    #print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
    #print("\n")

output_array = []
output_array.append(
    ['City', 'Point of Interest', 'Review', 'Adventure & Outdoors', 'Spiritual', 'Nature & Retreat', 'Isolated or Hippie', 'Heritage',
     'Travel & Learn', 'Social Tourism (Volunteer & Travel)', 'Nightlife & Events', 'Shopping', 'Compound score'])
test_review = original_test_data["Review"].values
poi = original_test_data["Place"].values
city = original_test_data["City"].values
for index in range(0, len(test_review)):
    row = []
    row.append(city[index])
    row.append(poi[index])
    row.append(test_review[index])
    score = analyser.polarity_scores(test_review[index])["compound"]
    norm_score = (score +1)/2
    for arr in arrs:
        row.append(arr[index])

    row.append(norm_score)

    output_array.append(row)

# print(output_array[1])

## Scoring

def formula_1(sum, total):
    return sum/total

def formula_2(n,sum,total):
    return (n*sum)/total*total

scores = {}

i = 1
while i<len(output_array):
    if output_array[i][1] not in scores.keys():
        scores[output_array[i][1]] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    j = 3
    while j < 12:
        if output_array[i][j] == 1:
            scores[output_array[i][1]][j-3] += output_array[i][12]
        j += 1
    scores[output_array[i][1]][9] += 1
    i += 1

with open('output.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(output_array)


for key in scores.keys():
    print(key + ":")
    for i in range(9):
        print(output_array[0][i+3], ":" + str(formula_1(scores[key][i], scores[key][9])))
    print(scores[key])
    print()

print(scores["Kunjapuri Devi Temple"])
print(scores["Agra Fort"])
