# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 11:47:56 2024

@author: zameer
"""
#import the file
import pandas as pd
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

#logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X,Y)


Y_pred = logreg.predict(X)
Y_pred
Y

#metrics
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score
cm = confusion_matrix(Y,Y_pred)
cm

ac = accuracy_score(Y,Y_pred)
print("Accuracy score:",ac.round(3))

rs = recall_score(Y,Y_pred)
print("Sensitivity score:",rs.round(3))

TN = cm[0,0]
FP = cm[0,1]
TNR = TN/(TN + FP)
print("Specificity score:",TNR.round(3))

#=========================================================
#LR3



from sklearn.metrics import precision_score,f1_score
print("precision_score:",precision_score(Y,Y_pred).round(3))
print("F1 score:",f1_score(Y,Y_pred).round(3))

df["Y_predected_prob"] = logreg.predict_proba(SS_X)[:,1]

df.head()

from sklearn.metrics import roc_curve,roc_auc_score

fpr,tpr,dummy = roc_curve(Y,df["Y_predected_prob"])

import matplotlib.pyplot as plt
plt.scatter(fpr,tpr)
plt.plot(fpr,tpr,color='red')
plt.xlabel("True positive Rate")
plt.ylabel("False positive Rate")
plt.show()

print("Area under curve:",roc_auc_score(Y,df["Y_predected_prob"]).round(2))















