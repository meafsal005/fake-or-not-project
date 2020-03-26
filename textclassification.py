import tweepy
auth =tweepy.OAuthHandler("x4yJNIGwJBspKCD3jXps0udJm","Xzi5PvN4rHvqBr6kPgcUCiSRDLLGLFJDFRZ1gPAvXxv5kvhyZZ")
auth.set_access_token("2491739339-q9jEkfotc22ETdAKFGNWiKLY7NCAtmy5cMbbMXI","UqdiqHDl1h5czl9jxpgDZQezm9mw88hA3qzX3Q72yHTA1")
api= tweepy.API(auth)
user = api.get_user('BJP4India')
tweet=user.status
#Loading the data set - training data.
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)


# In[4]:

# You can check the target names (categories) and some data files by following commands.
print(twenty_train.target_names) #prints all the categories


# In[5]:

print("\n".join(twenty_train.data[0].split("\n")[:3])) #prints first line of the first data file


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


# In[9]:

# Machine Learning
# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)


# In[14]:

# Building a pipeline: We can write less code and do all of the above, by building a pipeline as follows:
# The names ‘vect’ , ‘tfidf’ and ‘clf’ are arbitrary but will be used later.
# We will be using the 'text_clf' going forward.
from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

text_clf = text_clf.fit(twenty_train.data, twenty_train.target)


# In[15]:

# Performance of NB Classifier
import numpy as np
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted = text_clf.predict(twenty_test.data)
a=np.mean(predicted == twenty_test.target)


# Training Support Vector Machines - SVM and calculating its performance

from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42))])

text_clf_svm = text_clf_svm.fit(twenty_train.data, twenty_train.target)
predicted_svm = text_clf_svm.predict(twenty_test.data)
w=np.mean(predicted_svm == twenty_test.target)
print (w)
input=user.description
input=[input]
pr=text_clf_svm.predict(input)
res = str(pr)[1:-1]
print("belongs to the category : "+twenty_train.target_names[int(res)])

