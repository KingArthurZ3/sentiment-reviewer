import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import string
from sklearn.metrics import confusion_matrix, classification_report


yelp = pd.read_csv('./data/yelp.csv')

# grab only 1 and 5 stars reviews because we only need to know what reviews are bad or good
yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]

X = yelp_class['text']
y = yelp_class['stars']

def text_process(text):
    '''
    Takes in a string of text, then
    1. Removes all punctuation and stopwords
    3. Returns the clean text as a list of words
    '''
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

#transforms the data into a vector via z box of words model
bow_transformer = CountVectorizer(analyzer=text_process).fit(X)

X = bow_transformer.transform(X)

# Percentage of non-zero values
density = (100.0 * X.nnz / (X.shape[0] * X.shape[1]))

#split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

nb = MultinomialNB()
nb.fit(X_train, y_train)

preds = nb.predict(X_test)

print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))