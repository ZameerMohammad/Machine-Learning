# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:51:31 2024

@author: zameer
"""

import numpy as np 
import pandas as pd
d1 = {'id':[101,102,103,104,105],
      'age':[24,25,28,np.nan,29],
      'weight':[75,70,68,82,np.nan]}

d1
df1 = pd.DataFrame(d1)
df1

df1.isnull().sum()

#re-fill the blanks with mean / median
df1['age'].mean()
df1['age'].fillna(value=int(df1['age'].mean()),inplace=True)
df1
#------------------------------------------

d2 = {'id':[106,107,108,109,110],
      'age':[55,65,27,88,33],
      'weight':[77,78,58,92,22]}

d2
df2 = pd.DataFrame(d2)
df2

#------------------------------------
#concatinating - rows
df3 = pd.concat([df1,df2],ignore_index = True)
df3

#-===================================
d3 = {'Gender' : ["M","F","M","F","M","F","M","F","M","F"],
      'Height':[185,555,666,222,666,444,123,555,999,111]}

d3
df4 = pd.DataFrame(d3)
df4
#-----------------------------------------

#concatinating columns
df5 = pd.concat([df3,df4],axis=1)
df5

#------------------------------------------
df5['weight'].mean()

#group by 
df5.groupby(by='Gender').size()
df5.groupby(by='Gender').mean()# by default all numerical variables
df5.groupby(by='Gender').mean()['age']#by selected va
df5.groupby(by='Gender').mean()[['age','Height']]


#=====================================================

df = pd.read_csv("market_3.csv")
df

df.groupby(by='Region').size()

df6 = df.groupby(by='Region').mean()[['Sales','Returns']]
df6

#df6.iloc[1:2]

df6 = df.groupby(by='Region').mean()[['Sales','Returns']]
df6

df7 = df.groupby(by='Region').mean()
df7

df[df['Region'] == 'Asia']

df[df['Region'] == 'Asia'].groupby(by='Region').mean()[['Sales','Returns']]

#===============================================================

df.groupby(by='Product').size()

df.groupby(by='Product').mean()

df.groupby(by='Product').mean()[['Sales','Returns']]


#histogram
df.dtypes

df['Sales'].hist()
df['Sales'].skew()


#scatter plot
import matplotlib.pyplot as plt
plt.scatter(df['Sales'],df['Returns'],color='green')
plt.show()

#inventory , returns











