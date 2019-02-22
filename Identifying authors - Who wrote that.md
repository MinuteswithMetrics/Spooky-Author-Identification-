```
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

train_df.shape

# check the class distribution for the author label in train_df?
train_df['author'].value_counts()

# compute the text length for the rows and record these
train_df['text_length'] = train_df['text'].apply(lambda x: len(str(x).split()))

# number of text length that are greater than 100, 150 & 200
G100 = sum(i > 100 for i in train_df['text_length'])
G150 = sum(i > 150 for i in train_df['text_length'])
G200 = sum(i > 200 for i in train_df['text_length'])
print('Text length greater than 100, 150 & 200 are ',G100,',',G150,'&',G200, ' respectively.')
print('In percentages, they are %.2f, %.2f & %.2f' %(G100/len(train_df)*100, 
      G150/len(train_df)*100, G200/len(train_df)*100))
      
EAP = train_df[train_df['author'] =='EAP']['text_length']

MWS = train_df[train_df['author'] == 'MWS']['text_length']

HPL = train_df[train_df['author'] == 'HPL']['text_length']

test_df.shape

# examine the text length in test_df and record these
test_df['text_length'] = test_df['text'].apply(lambda x: len(str(x).split()))

# number of text length that are greater than 100, 150 & 200
G100 = sum(i > 100 for i in test_df['text_length'])
G150 = sum(i > 150 for i in test_df['text_length'])
G200 = sum(i > 200 for i in test_df['text_length'])
print('Text length greater than 100, 150 & 200 are ',G100,',',G150,'&',G200, ' respectively.')
print('In percentages, they are {:.2f}, {:.2f} & {:.2f}'.format(G100/len(test_df)*100, 
      G150/len(test_df)*100, G200/len(test_df)*100))
      
# convert author labels into numerical variables
train_df['author_num'] = train_df.author.map({'EAP':0, 'HPL':1, 'MWS':2})
# Check conversion for first 5 rows

X = train_df['text']
y = train_df['author_num']
print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# examine the class distribution in y_train and y_test
print(y_train.value_counts())
print(y_test.value_counts())

# import and instantiate CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(lowercase=False, token_pattern=r'(?u)\b\w+\b|\,|\.|\;|\:')
# vect = CountVectorizer(lowercase=False, token_pattern=r'(?u)\b\w+\b|\,|\.|\?|\;|\:|\!|\'')
vect

# learn the vocabulary in the training data, then use it to create a document-term matrix
X_train_dtm = vect.fit_transform(X_train)
# examine the document-term matrix created from X_train
X_train_dtm

# transform the test data using the earlier fitted vocabulary, into a document-term matrix
X_test_dtm = vect.transform(X_test)
# examine the document-term matrix from X_test
X_test_dtm

# import and instantiate the Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb

# train the model using X_train_dtm & y_train
nb.fit(X_train_dtm, y_train)

# make author (class) predictions for X_test_dtm
y_pred_test = nb.predict(X_test_dtm)

# compute the accuracy of the predictions with y_test
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_test)

# compute the accuracy of training data predictions
y_pred_train = nb.predict(X_train_dtm)
metrics.accuracy_score(y_train, y_pred_train)

#look at the confusion matrix for y_test
metrics.confusion_matrix(y_test, y_pred_test)

# calculate predicted probabilities for X_test_dtm
y_pred_prob = nb.predict_proba(X_test_dtm)
y_pred_prob[:10]

# compute the log loss number
metrics.log_loss(y_test, y_pred_prob)

# import and instantiate the Logistic Regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg

# train the model using X_train_dtm and y_train
logreg.fit(X_train_dtm, y_train)

# make class predictions for X_test_dtm
y_pred_test = logreg.predict(X_test_dtm)

# compute the accuracy of the predictions
metrics.accuracy_score(y_test, y_pred_test)

# compute the accuracy of predictions with the training data
y_pred_train = logreg.predict(X_train_dtm)
metrics.accuracy_score(y_train, y_pred_train)

# look at the confusion matrix for y_test
metrics.confusion_matrix(y_test, y_pred_test)

# compute the predicted probabilities for X_test_dtm
y_pred_prob = logreg.predict_proba(X_test_dtm)
y_pred_prob[:10]

# compute the log loss number
metrics.log_loss(y_test, y_pred_prob)

# Learn the vocabulary in the entire training data, and create the document-term matrix
X_dtm = vect.fit_transform(X)
# Examine the document-term matrix created from X_train
X_dtm

# Train the Logistic Regression model using X_dtm & y
logreg.fit(X_dtm, y)

# Compute the accuracy of training data predictions
y_pred_train = logreg.predict(X_dtm)
metrics.accuracy_score(y, y_pred_train)
```

# transform the test data using the earlier fitted vocabulary, into a document-term matrix
test_dtm = vect.transform(test_df['text'])
# examine the document-term matrix from X_test
test_dtm

# make author (class) predictions for test_dtm
LR_y_pred = logreg.predict(test_dtm)
print(LR_y_pred)

# calculate predicted probabilities for test_dtm
LR_y_pred_prob = logreg.predict_proba(test_dtm)
LR_y_pred_prob[:10]

nb.fit(X_dtm, y)

# compute the accuracy of training data predictions
y_pred_train = nb.predict(X_dtm)
metrics.accuracy_score(y, y_pred_train)

# make author (class) predictions for test_dtm
NB_y_pred = nb.predict(test_dtm)
print(NB_y_pred)

# calculate predicted probablilities for test_dtm
NB_y_pred_prob = nb.predict_proba(test_dtm)
NB_y_pred_prob[:10]

y_pred_prob = (LR_y_pred_prob + NB_y_pred_prob)/2
y_pred_prob[:10]

result = pd.DataFrame(y_pred_prob, columns=['EAP','HPL','MWS'])
result.insert(0, 'id', test_df['id'])

# Generate submission file in csv format
result.to_csv('green_submission_13.csv', index=False, float_format='%.20f')
