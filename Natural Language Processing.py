# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score

# %% [markdown]
# # Data Preprocessing

# %%
# Data set was already split into training and testing sets

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# %%
# Shape of training data 
df_train.shape

# %%
# Shape of testing data 
df_test.shape

# %%
# Describing training dataset
df_train.describe()

# %%
# Describing testing dataset
df_test.describe()

# %%
# Top 5 of training set
df_train.head()

# %%
#  Function to calculate total of missing values in dataset
def missing_values(df):
    print("Number of records with missing location:",df.location.isna().sum())
    print("Number of records with missing keywords:",df.keyword.isna().sum())
    

# %%
# Checking missing values of training set
missing_values(df_train)

# %%
# Missing values of testing set
missing_values(df_test)

# %%
# Check for keywords count
keywords = df_train['keyword'].value_counts()
print(keywords.head())

# %%
# Check location counts
locations = df_train['location'].value_counts()
print(locations.head())

# %%
#Create barchart for locations in train set using seaborn
sns.barplot(y=df_train['location'].value_counts()[:10].index,x=df_train['location'].value_counts()[:10],
            orient='h')

# %%
#Create barchart for locations in test set using seaborn
sns.barplot(y=df_test['location'].value_counts()[:10].index,x=df_test['location'].value_counts()[:10],
            orient='h')

# %% [markdown]
# Group By

# %%
# Groupby 
df_train.groupby('target').count()

# %%
# Group the tweets by disaster or not for the training data
grouped = df_train.groupby(['target'])['text'].count()

# plot the same as bar chart
grouped.plot(kind='bar')
plt.title('Disaster Tweet frequency chart for training data')
plt.xlabel('Disaster or not')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# Tweet Lengths

# %%
# Calculate tweet lengths
df_train['length'] = df_train['text'].apply(lambda x : len(x))
df_train.head() # Check new columns

# %%
# Visualization of Tweet Lengths in training data
# Code source: https://seaborn.pydata.org/generated/seaborn.displot.html

sns.displot(data=df_train['length'], kde=True)

# %%
# Dropping unnecessary columns
df_train = df_train.drop(columns=['keyword', 'location', 'length'])

# %% [markdown]
# # Text Vectorization

# %%
# Import string
import string

# Import nltk (Natural Language Toolkit)
import nltk
nltk.download('stopwords')

# NLTK packages
from nltk.corpus import stopwords
from nltk import PorterStemmer as Stemmer

# Code source for text preprocessing: 
# https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/

def preprocess(text):
    # lowercase 
    text = text.lower()
    
    # Remove punctuation
    text = ''.join([t for t in text if t not in string.punctuation])
    
    # Removing stopwords since they do not add value to this analysis
    # Code source: https://pythonprogramming.net/stop-words-nltk-tutorial/
    text = [t for t in text.split() if t not in stopwords.words('english')]
    
    # Stemming is used to reducing words to their root 
    # Code source: https://www.kaggle.com/code/sudalairajkumar/getting-started-with-text-preprocessing
    stemmer = Stemmer()
    text = [stemmer.stem(t) for t in text]
    
    # return text 
    return text

# The function above is used to normalize and tokenize 
# texts that were found in the 'Text' column. 
# we cleaned the 'Text' column as much as we could by 
# using the NLTK (Natural Language Toolkit) library 
# that we found in their documentation (https://www.nltk.org/). 
# By cleaning this up, we are able reduce 
# the size of the vocab when we input into our machine learning model. 

# %%
# Test with dataset, the first 20 rows
df_train['text'][:20].apply(preprocess)

# %%
# Test with dataset, the first 20 rows
df_test['text'][:20].apply(preprocess)

# %%
# Fit transform
TFID = TfidfVectorizer(analyzer=preprocess)
fit = TFID.fit_transform(df_train['text'])
fits = TFID.fit_transform(df_test['text'])

# %%
# Checking values
content = df_train.iloc[50]['text'] # Randomly chose 50th index
print(content) # Print message

# %%
# Assigning texts to vectors

# Code source:
# https://github.com/scikit-learn/scikit-learn/blob/8c9c1f27b/sklearn/feature_extraction/text.py#L1470

# Code source 2: 
# https://www.kaggle.com/code/jeffysonar/spam-filter-using-naive-bayes-classifier/notebook

# Inputing "content" into transform function and adding to an array
tfid = TFID.transform(['text']).toarray()[0]

print('index\tidf\ttfidf\tterm') # Print in this order

# Loop function to assign different values to its term. 
for i in range(len(tfid)):
    if tfid[i] != 0:
        print(i, format(TFID
                        .idf_[i], '.5f'), format(tfid[i], 
                                                 '.5f'), 
                        TFID.get_feature_names_out()[i],sep='\t')

# %% [markdown]
# # Logistic Regression

# %%
# Check size of text column
training_texts = df_train['text']

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(training_texts)
y_train = df_train['target']
X_test = vectorizer.transform(df_test['text'])

print(X_train.size)
print(y_train.size)

# %%
# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# %%
# Logistic regression model for predicting diaster in tweets 
log = LogisticRegression()

# Train the model on the training data
log.fit(X_train, y_train)

# Make predictions on the test data
log_pred = log.predict(X_test)
print(classification_report(y_test, log_pred))

# %% [markdown]
# # Multinomial NB

# %%
mnb = MultinomialNB()

# Fitting the model
mnb.fit(X_train, y_train)

# Evaluate the Multinomial NB model on the testing set
nb_pred = mnb.predict(X_test)
print(classification_report(y_test, nb_pred))

# %% [markdown]
# # SVM

# %%
from sklearn.svm import SVC
from sklearn.metrics import silhouette_score, accuracy_score, classification_report

# Building and training SVM model
svm = SVC()

# Fitting the model
svm.fit(X_train, y_train)

# Evaluate the SVM model on the testing set
svm_pred = svm.predict(X_test)
print(classification_report(y_test, svm_pred))

# %% [markdown]
# # Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier

# Building Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the classifier
rf_pred = rf.predict(X_test)
print(classification_report(y_test, rf_pred))

# %% [markdown]
# # Neural Networks

# %%
from sklearn.preprocessing import StandardScaler

# Standardize the  dataset
scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout
from keras.models import Sequential


# %%
# Tokenizer documentation: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer
# pad_sequences documentation: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences
# Sequential model documentation: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# Embedding layer documentation: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
# Dense layer documentation: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
# Flatten layer documentation: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten
# Dropout layer documentation: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout

# Preprocess the text data
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(df_train['text'])
tokenizer.fit_on_texts(df_test['text'])
train_sequences = tokenizer.texts_to_sequences(df_train['text'])
train_padded = pad_sequences(train_sequences, padding='post', truncating='post')
test_sequences = tokenizer.texts_to_sequences(df_test['text'])
test_padded = pad_sequences(test_sequences, padding='post', truncating='post')


# %%
# create model
model = Sequential()
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Build model
model.build(input_shape=(None, 21637)) 

 # Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# %%
# Converting to  numpy arras to be compatible with keras
X_test = np.array(X_test)
y_test = np.array(y_test)

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)
print('Test loss:', accuracy[0])
print('Test accuracy:', accuracy[1])


