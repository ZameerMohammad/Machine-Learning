# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:18:11 2024

@author: zameer
"""

#import the file
import pandas as pd
import numpy as np
df = pd.read_csv("breast_cancer.csv")
df
list(df)
df.dtypes
df.shape


#split as X and Y variables
X = df.iloc[:,1:9]
Y = df['Class']
X

#data transformation
#lable encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
Y=LE.fit_transform(Y)
Y

#standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X =  SS.fit_transform(X)
SS_X = pd.DataFrame(SS_X)

# Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=11)

X_train.shape
X_test.shape

#logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(X_train,Y_train)

Y_pred_train = logreg.predict(X_train)
Y_pred_test = logreg.predict(X_test)

#metrics
from sklearn.metrics import accuracy_score
ac_train = accuracy_score(Y_train,Y_pred_train)
ac_test = accuracy_score(Y_test,Y_pred_test)

print("Training Accuracy:", ac_train.round(2))
print("Test Accuracy:", ac_test.round(2))

#cross validation

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    logreg.fit(X_train,Y_train)
    Y_pred_train = logreg.predict(X_train)
    Y_pred_test = logreg.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
    
# pd. DtaFrame(training_accuracy).hist()
k1 = pd.DataFrame(test_accuracy)
k1.skew()
k1.mean()
print("Cross validation training results:",k1.mean().round(2))

#pd. DtaFrame(training_accuracy).hist()
k2 = pd.DataFrame(test_accuracy)
k2.skew()
k2.mean()
print("Cross validation training results:",k2.mean().round(2))


# K-FLOD Cross validation

from sklearn.model_selection import KFold
KFold = KFold(n_splits=5)

'''
for i in range(0,30):
    print(i)
for train_index,test_index in KFold.split(range(0,30)):
    print(test_index)
    print(train_index)
'''

training_accuracy = []
test_accuracy = []
for train_index,test_index in KFold.split(SS_X):
    X_train,X_test = SS_X.iloc[train_index],SS_X.iloc[test_index]
    Y_train,Y_test = Y[train_index],Y[test_index]
    logreg.fit(X_train,Y_train)
    Y_pred_train = logreg.predict(X_train)
    Y_pred_test = logreg.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
    
    
print("K-Fold Cross validation training results:",np.mean(training_accuracy).round(2))
print("K-Fold Cross validation training results:",np.mean(test_accuracy).round(2))

#=============KNN Classifer======================#


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=5)

KNN.fit(X_train,Y_train)

Y_pred_train = KNN.predict(X_train)
Y_pred_test = KNN.predict(X_test)

#metrics
from sklearn.metrics import accuracy_score
ac_train = accuracy_score(Y_train,Y_pred_train)
ac_test = accuracy_score(Y_test,Y_pred_test)

print("Training Accuracy:", ac_train.round(2))
print("Test Accuracy:", ac_test.round(2))

#cross validation

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    KNN.fit(X_train,Y_train)
    Y_pred_train = KNN.predict(X_train)
    Y_pred_test = KNN.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
    
# pd. DtaFrame(training_accuracy).hist()
k1 = pd.DataFrame(test_accuracy)
print("Cross validation training results:",k1.mean().round(2))

#pd. DtaFrame(training_accuracy).hist()
k2 = pd.DataFrame(test_accuracy)
print("Cross validation training results:",k2.mean().round(2))





















