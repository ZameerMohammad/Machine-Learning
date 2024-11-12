# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:17:11 2024

@author: zameer
"""

#import the file
import pandas as pd
import numpy as np
df = pd.read_csv("createdata.csv")
df
list(df)
df.dtypes
df.shape

#split as X and Y variables
X = df.iloc[:,0:2]
Y = df['Y']
X

# Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=11)

#support vector machine
from sklearn.svm import SVC
svclass = SVC(C=1.0,kernel='linear') 

svclass.fit(X_train,Y_train)

Y_pred_train = svclass.predict(X_train)
Y_pred_test = svclass.predict(X_test)

#metrics
from sklearn.metrics import accuracy_score
ac_train = accuracy_score(Y_train,Y_pred_train)
ac_test = accuracy_score(Y_test,Y_pred_test)

print("Training Accuracy:", ac_train.round(2))
print("Test Accuracy:", ac_test.round(2))

#cross validation

from sklearn.svm import SVC
svclass = SVC(C=1.0,kernel='linear') 

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    svclass.fit(X_train,Y_train)
    Y_pred_train = svclass.predict(X_train)
    Y_pred_test = svclass.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
    
k1 = pd.DataFrame(test_accuracy)
print("Cross validation training results:",k1.mean().round(2))
k2 = pd.DataFrame(test_accuracy)
print("Cross validation training results:",k2.mean().round(2))


#======================================================
#data visualization
pip install mlxtend
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X=X.values,
                      y=Y.values,          
                      clf=svclass,
                      legend=4)

#=================================================
#polynomial

from sklearn.svm import SVC
svclass = SVC(C=1.0,kernel='poly',degree=2) 

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    svclass.fit(X_train,Y_train)
    Y_pred_train = svclass.predict(X_train)
    Y_pred_test = svclass.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
    
k1 = pd.DataFrame(test_accuracy)
print("Cross validation training results:",k1.mean().round(2))
k2 = pd.DataFrame(test_accuracy)
print("Cross validation training results:",k2.mean().round(2))

from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X=X.values,
                      y=Y.values,          
                      clf=svclass,
                      legend=4)



#====================================================
#rbf

from sklearn.svm import SVC
svclass = SVC(C=1.0,kernel='rbf') 

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    svclass.fit(X_train,Y_train)
    Y_pred_train = svclass.predict(X_train)
    Y_pred_test = svclass.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
    
k1 = pd.DataFrame(test_accuracy)
print("Cross validation training results:",k1.mean().round(2))
k2 = pd.DataFrame(test_accuracy)
print("Cross validation training results:",k2.mean().round(2))

from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X=X.values,
                      y=Y.values,          
                      clf=svclass,
                      legend=4)



















