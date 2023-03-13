# Natural-Language-Process
This repo contains a project overview and source code for the final project of IMT 574 - Data Science II for the University of Washington Information School, Master of Science in Information Science. 

## Introduction
It is estimated that 80% of data exists in unstructured form, and much of that is text. Methods such as NLP can harness that data and help us make sense of it. Twitter has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. 
Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).
But, it’s not always clear whether a person’s words are actually announcing a disaster.


In this project the objective is the classification of text in tweets to determine if the tweet is about a disaster or potential emergency or not.

Computers can now generate text, translate automatically from one language to another, analyze comments, label words in sentences, and much more.
Perhaps the most widely and practically useful application of NLP is classification which can be used for:
- Sentiment analysis 
- Are product review positive or negative?
- Author identification 
- Which author most likely wrote some document?
- Legal discovery
- Which documents are in scope for a trial?
- Organizing documents by topic
- Triaging inbound emails
- ...and much more!

# The Data

The data is in the form of a collection of tweets from Kaggle. The data files include 10,876 tweets split into a training set and a testing set. The training set has been previously classified.

The focus of the project is on the variable of the text in the tweet, which is to be classified as a true disaster or not a disaster at all.

The text is raw, unstructured data that requires cleaning and preprocessing before NLP can be applied.


# Data Processing
- Drop irrelevant columns such as: 'keywords', 'location'
- Convert text to lowercase
- Remove stopwords
- Remove punctuation
- Tokenize text
- Remove duplicates
- Remove non-alphanumeric characters

# Models Selection
- Naive Bayes: It is the most simple and efficient classification algorithm for text classification. Works really well in small datasets. Naive Bayes assumes that the features (words) are independent, which can be a limitation in some cases.
- Logistic Regression: Another classification algorithm for sparse and dense feature representation. It is useful for identifying which features are most important for a given classification task.    
- Support Vector Machines (SVMs): Classification algorithm that tries to separate the data into different classes. It works really well  for tasks such as text classification and named entity recognition. It is the most expensive model and may require careful tuning of the model 
- Random Forest: This is an ensemble learning algorithm that combines multiple decision trees to improve performance and reduce overfitting. They are relatively easy to use and can handle both categorical and continuous features.
- Neural Networks: Neural networks can be used for a wide range of NLP tasks, including text classification, machine translation, and natural language generation. They require large amounts of data and may be computationally expensive to train, but can achieve state-of-the-art performance on many tasks.






