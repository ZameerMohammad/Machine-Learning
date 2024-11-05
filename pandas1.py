# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:00:24 2024

@author: zameer
"""

import pandas as pd
df = pd.read_csv("nyc_weather.csv")
df

df.shape
df.dtypes

#acess the 

df.head()#first five rows
df.head(10)
df.tail()#last five rows

df["Temperature"]
df["Temperature"].min()
df["Temperature"].max()
df["Temperature"].mean()
df["Temperature"].median()
df["Temperature"].std()
df["Temperature"].var()
df["Temperature"].mode()
df["Temperature"].describe()

#columns inside the data
list(df)
df.columns

# accessing the columns with their names
df[['Temperature','Humidity','VisibilityMiles']]
df[['Temperature','Humidity','VisibilityMiles']].describe()

# access the columns with their columns positions
df.columns[[0,2,4]]# displaying columns names
df[df.columns[[0,2,4]]]# displaying the results of those columns

# how to drop the unwanted columns
list(df)

df.columns[[0,2,10]]# displaying columns names

#only for temporary we removed the columns
df.drop(df.columns[[0,2,10]], axis=1)
df.head()

#permanantely
df.drop(df.columns[[0,2,10]], axis=1,inplace=True)
df.head()

#####################################

#How to find the blanks
df.isnull().sum()

df["Events"].value_counts()

# filtering the data
df["Events"] == 'Rain'
df_rainy = df[df["Events"]=='Rain']


df_rainy
df_rainy.to_csv("rainy_file.csv")





























