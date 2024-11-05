# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:06:40 2024

@author: zameer
"""

#step1: import the file 
import numpy as np
import pandas as pd
df=pd.read_csv("Sales_cars_seats.csv")
df.shape
df.head

list(df)
df.dtypes

df_cont=df.drop(df.columns[[0,7,10,11]],axis=1)
df_cont.head()

#==================Standard scaler=======================
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()

SS_X = SS.fit_transform(df_cont)
SS_X
SS_X = pd.DataFrame(SS_X)
SS_X
list(df_cont)
SS_X.columns = list(df_cont)
SS_X.head()

#============min max Scaler========================
from sklearn.preprocessing import MinMaxScaler
MM = MinMaxScaler()

MM_X = SS.fit_transform(df_cont)
MM_X
MM_X = pd.DataFrame(MM_X)
MM_X
list(df_cont)
MM_X.columns = list(df_cont)
MM_X.head()

#==================Label encoder==================

df.dtypes 

df_cat = df[['ShelveLoc','Urban','US']]
df_cat.head()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

df_cat['ShelveLoc'] = LE.fit_transform(df_cat['ShelveLoc'])
df_cat['Urban'] = LE.fit_transform(df_cat['Urban'])
df_cat['US'] = LE.fit_transform(df_cat['US'])

df_cat.head()

#================

SS_X
df_cat
df_final = pd.concat([SS_X,df_cat],axis=1)
df_final
















