import tweepy
auth =tweepy.OAuthHandler("***")
auth.set_access_token("***")
api= tweepy.API(auth)
user = api.get_user('BJP4India')
tweet=user.status
#Loading the data set - training data.
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)


# In[6]:

# Extracting features from text files
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape


# In[7]:

# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape



# In[15]:

# Performance of NB Classifier
import numpy as np
from sklearn.pipeline import Pipeline
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)


# Training Support Vector Machines - SVM and calculating its performance

from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42))])

text_clf_svm = text_clf_svm.fit(twenty_train.data, twenty_train.target)

input=user.description
input=[input]
pr=text_clf_svm.predict(input)
res = str(pr)[1:-1]
print("belongs to the category : "+twenty_train.target_names[int(res)])

