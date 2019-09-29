import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import re
import spacy


def get_data(filename):
    data_frame = pd.read_csv(filename, delimiter = ',')
    labels = data_frame['label'].values
    tweets = data_frame['tweet'].values


    df_stratif = StratifiedShuffleSplit(n_splits = 1, test_size=0.2)

    for train_index, validation_index in df_stratif.split(tweets, labels):
        tweets_train, tweets_validation = tweets[train_index], tweets[validation_index]
        labels_train, labels_validation = labels[train_index], labels[validation_index]

    return tweets_train, labels_train, tweets_validation, labels_validation


def word_processing(sentence):
    results = ''

    for token in sentence:
        if len(token) > 3:
            results = results + " " + str(token)
    return results


doc_list1 = []
doc_list2 = []
tweets_train = []
tweets_validation = []

words_train, labels_train, words_validation, labels_validation = get_data('train.csv')

words_train = [re.sub('[0-9]+', '', i) for i in words_train]
tweets_validation = [re.sub('[0-9]+', '', i) for i in tweets_validation]

nlp = spacy.load('en_core_web_sm')

for i in words_train:
    i = str(i)
    doc1 = nlp(i)
    doc_list1.append(doc1)

for j in words_validation:
    j = str(j)
    doc2 = nlp(j)
    doc_list2.append(doc2)

for i in doc_list1:
    tweets_train.append(word_processing(i))

for j in doc_list2:
    tweets_validation.append(word_processing(j))

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
x_train = tfidf_vectorizer.fit_transform(tweets_train)
x_validation = tfidf_vectorizer.transform(tweets_validation)

model = MultinomialNB(alpha = 0.01)
model.fit(x_train, labels_train)

predictions_validation = model.predict(x_validation)
predictions_train = model.predict(x_train)

print(accuracy_score(labels_train, predictions_train))
print(classification_report(labels_train, predictions_train))
print(accuracy_score(labels_validation, predictions_validation))
print(classification_report(labels_validation, predictions_validation))

