# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:41:25 2024

@author: zameer
"""

from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor(learning_rate=0.1,
                                n_estimators=100)  


#import the file
import pandas as pd
import numpy as np
df = pd.read_csv("Boston.csv")
df
list(df)
df.dtypes
df.shape

#split as X and Y variables
X = df.iloc[:,1:14]
Y = df['medv']
X

#standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X =  SS.fit_transform(X)
SS_X = pd.DataFrame(SS_X)

# Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=11)

#Decision tree regression
from sklearn.tree import DecisionTreeRegressor
DT = DecisionTreeRegressor()

DT.fit(X_train,Y_train)

Y_pred_train = DT.predict(X_train)
Y_pred_test = DT.predict(X_test)


#metrics
from sklearn.metrics import mean_squared_error
train_error = np.sqrt(mean_squared_error(Y_train,Y_pred_train))
test_error = np.sqrt(mean_squared_error(Y_test,Y_pred_test))

print("Training Error:", train_error.round(2))
print("Test Error:", test_error.round(2)) 

#cross validation

training_error = []
test_error = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    DT.fit(X_train,Y_train)
    Y_pred_train = DT.predict(X_train)
    Y_pred_test = DT.predict(X_test)
    training_error.append(np.sqrt(mean_squared_error(Y_train,Y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(Y_test,Y_pred_test)))
    

print(" Cross validation training results:",np.mean(training_error).round(2))
print("Cross validation training results:",np.mean(test_error).round(2))


#==================
#bagging regressor

from sklearn.ensemble import BaggingRegressor
bag = BaggingRegressor(base_estimator=DT,n_estimators=100,
                       max_features=0.7,max_samples=0.6)

training_error = []
test_error = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    bag.fit(X_train,Y_train)
    Y_pred_train = bag.predict(X_train)
    Y_pred_test = bag.predict(X_test)
    training_error.append(np.sqrt(mean_squared_error(Y_train,Y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(Y_test,Y_pred_test)))
    

print(" Cross validation training results:",np.mean(training_error).round(2))
print("Cross validation training results:",np.mean(test_error).round(2))
print("variance:",(np.mean(test_error)-np.mean(training_error).round(2)))

#=======================
#random forest regressor

from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators=100,
                       max_features=0.7,max_samples=0.6)

training_error = []
test_error = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    RFR.fit(X_train,Y_train)
    Y_pred_train = RFR.predict(X_train)
    Y_pred_test = RFR.predict(X_test)
    training_error.append(np.sqrt(mean_squared_error(Y_train,Y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(Y_test,Y_pred_test)))
    

print(" Cross validation training results:",np.mean(training_error).round(2))
print("Cross validation training results:",np.mean(test_error).round(2))
print("variance:",(np.mean(test_error)-np.mean(training_error).round(2)))

#===========================
#gradient boosting regressor

from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor(learning_rate=0.1,
                                n_estimators=100)
                                 

training_error = []
test_error = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    GBR.fit(X_train,Y_train)
    Y_pred_train = GBR.predict(X_train)
    Y_pred_test = GBR.predict(X_test)
    training_error.append(np.sqrt(mean_squared_error(Y_train,Y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(Y_test,Y_pred_test)))
    

print(" Cross validation training results:",np.mean(training_error).round(2))
print("Cross validation training results:",np.mean(test_error).round(2))
print("variance:",(np.mean(test_error)-np.mean(training_error).round(2)))


#===========================
# ADA boosting regressor

from sklearn.ensemble import AdaBoostRegressor
ABR = AdaBoostRegressor(base_estimator=DT,learning_rate=1,
                                n_estimators=100)
                                 

training_error = []
test_error = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    ABR.fit(X_train,Y_train)
    Y_pred_train = ABR.predict(X_train)
    Y_pred_test = ABR.predict(X_test)
    training_error.append(np.sqrt(mean_squared_error(Y_train,Y_pred_train)))
    test_error.append(np.sqrt(mean_squared_error(Y_test,Y_pred_test)))
    

print(" Cross validation training results:",np.mean(training_error).round(2))
print("Cross validation training results:",np.mean(test_error).round(2))
print("variance:",(np.mean(test_error)-np.mean(training_error).round(2)))
























