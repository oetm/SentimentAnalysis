from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
import re

def get_data(filename):
    data_frame = pd.read_csv(filename, delimiter = ',')
    labels = data_frame['label'].values
    tweets = data_frame['tweet'].values

    df_stratif = StratifiedShuffleSplit(n_splits = 1, test_size=0.2)

    for train_index, validation_index in df_stratif.split(tweets, labels):
        tweets_train, tweets_validation = tweets[train_index], tweets[validation_index]
        labels_train, labels_validation = labels[train_index], labels[validation_index]

    return tweets_train, labels_train, tweets_validation, labels_validation

count_vectorizer = CountVectorizer(lowercase=True, analyzer = 'word', stop_words='english')

tweets_train, labels_train, tweets_validation, labels_validation = get_data('train.csv')

tweets_train = [re.sub('[0-9]+', '', i) for i in tweets_train]
tweets_validation = [re.sub('[0-9]+', '', i) for i in tweets_validation]

x_train = count_vectorizer.fit_transform(tweets_train)
x_validation = count_vectorizer.transform(tweets_validation)

model = SVC(C = 0.8, kernel = 'linear')
model.fit(x_train, labels_train)

predictions_validation = model.predict(x_validation)
predictions_train = model.predict(x_train)

print(accuracy_score(labels_train, predictions_train))
print(classification_report(labels_train, predictions_train))
print(accuracy_score(labels_validation, predictions_validation))
print(classification_report(labels_validation, predictions_validation))