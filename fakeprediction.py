# Load dataset
import csv
import datetime
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
auth =tweepy.OAuthHandler("****")
auth.set_access_token("****")
api= tweepy.API(auth)
names = ['twtCount', 'engRatio','frndcount','biolen','likecount','verified','default','class']
ds = pandas.read_csv("Desktop/dataset_original.csv", names=names)
print("shape: {}".format(ds.shape))
array=ds.values
X=array[:,0:7]
Y=array[:,7]
X_train,X_validation,Y_train,Y_validation = train_test_split(X,Y,test_size=0.30,random_state=None)
print("x shape: {}".format(X_train.shape))
print("y shape: {}".format(Y_train.shape))
print("x test shape: {}".format(X_validation.shape))
print("y test shape: {}".format(Y_validation.shape))
print("x_train:{}".format(X_train))
print("X_test:{}".format(X_validation))

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
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
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
lda=LinearDiscriminantAnalysis()
lda.fit(X_train,Y_train)
uname='iSachinDeshwal'
user = api.get_user(uname)
tweet=user.status
if (user.verified)== True:
	a=1
else:
	a=0
if (user.default_profile)== True:
	b=0
else:
	b=1
endDate =   datetime.datetime(2020, 4, 13, 0, 0, 0)
startDate = datetime.datetime(2020, 4, 7, 0, 0, 0)

tweets = []
tmpTweets = api.user_timeline(uname)
for tweet in tmpTweets:
    if (tweet.created_at < endDate and tweet.created_at > endDate): tweets.append(tweet.favorite_count)
while (tmpTweets[-1].created_at > startDate):
    tmpTweets = api.user_timeline(uname, max_id = tmpTweets[-1].id)
    for tweet in tmpTweets:
        if tweet.created_at < endDate and tweet.created_at > startDate:
            tweets.append(tweet.favorite_count)
tweets.sort()
X_new=np.array([[user.statuses_count,(tweets[-1]/user.followers_count)*1000,user.friends_count,len(user.description),user.favourites_count,a,b]])
print(X_new)
prediction = lda.predict(X_new)
print("prediction: {}" .format(prediction))
