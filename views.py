from django.shortcuts import render
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User, auth
from sklearn.feature_extraction.text import CountVectorizer


# Create your views here.
# Load dataset
import csv
import pandas
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
import tweepy
import datetime


def home(request):
    return render(request, 'home.html')


def fakeornot(request):
    auth = tweepy.OAuthHandler(
        "x4yJNIGwJBspKCD3jXps0udJm", "Xzi5PvN4rHvqBr6kPgcUCiSRDLLGLFJDFRZ1gPAvXxv5kvhyZZ")
    auth.set_access_token("2491739339-q9jEkfotc22ETdAKFGNWiKLY7NCAtmy5cMbbMXI",
                          "UqdiqHDl1h5czl9jxpgDZQezm9mw88hA3qzX3Q72yHTA1")
    api = tweepy.API(auth)
    names = ['twtCount', 'engRatio', 'frndcount', 'biolen',
        'likecount', 'verified', 'default', 'class']
    ds = pandas.read_csv("static/dataset_original.csv", names=names)
    print("shape: {}".format(ds.shape))
    array = ds.values
    X = array[:, 0:7]
    Y = array[:, 7]
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, Y, test_size=0.30, random_state=None)
    print("x shape: {}".format(X_train.shape))
    print("y shape: {}".format(Y_train.shape))
    print("x test shape: {}".format(X_validation.shape))
    print("y test shape: {}".format(Y_validation.shape))
    print("x_train:{}".format(X_train))
    print("X_test:{}".format(X_validation))

# Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression(
        solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
    results = []
    names = []
    for name, model in models:
	    kfold = StratifiedKFold(n_splits=5, random_state=None)
	    cv_results = cross_val_score(
	        model, X_train, Y_train, cv=kfold, scoring='accuracy')
	    results.append(cv_results)
	    names.append(name)
	    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, Y_train)
    p = request.POST['name1']
    user = api.get_user(p)
    tweet = user.status
    endDate = datetime.datetime(2020, 5, 14, 0, 0, 0)
    startDate = datetime.datetime(2020, 5, 8, 0, 0, 0)
    tweets = []
    tmpTweets = api.user_timeline(p)
    for tweet in tmpTweets:
	    if (tweet.created_at < endDate and tweet.created_at >
	        endDate): tweets.append(tweet.favorite_count)
    while (tmpTweets[-1].created_at > startDate):
	    tmpTweets = api.user_timeline(p, max_id=tmpTweets[-1].id)
	    for tweet in tmpTweets:
	        if tweet.created_at < endDate and tweet.created_at > startDate:
	            tweets.append(tweet.favorite_count)

    if (user.verified) == True:
	    a = 1
    else:
	    a = 0
    if (user.default_profile) == True:
	    b = 0
    else:
	    b = 1

    if(len(tweets)==0):
	    t=user.status.favorite_count
    else:
	    tweets.sort()
	    t=tweets[-1]
    X_new=np.array([[user.statuses_count,(t/user.followers_count)*1000,user.friends_count,len(user.description),user.favourites_count,a,b]])
    prediction = lda.predict(X_new)
    q=prediction
    print(prediction)
    if q==['not']:
	    v=0
	    return render(request,'not.html',{'result':'This is a genuine twitter account','l':p})
    else:
	    v=1



	    auth =tweepy.OAuthHandler("x4yJNIGwJBspKCD3jXps0udJm","Xzi5PvN4rHvqBr6kPgcUCiSRDLLGLFJDFRZ1gPAvXxv5kvhyZZ")
	    auth.set_access_token("2491739339-q9jEkfotc22ETdAKFGNWiKLY7NCAtmy5cMbbMXI","UqdiqHDl1h5czl9jxpgDZQezm9mw88hA3qzX3Q72yHTA1")
	    api= tweepy.API(auth)
	    user = api.get_user(p)
	    tweet=user.status
     # Loading the data set - training data.
	    from sklearn.datasets import fetch_20newsgroups
	    twenty_train = fetch_20newsgroups(subset='train', shuffle=True)


# In[6]:

# Extracting features from text files

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
	    import numpy as nv
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
	    return render(request,'not2.html',{'result':twenty_train.target_names[int(res)]})


    
