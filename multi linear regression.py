# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:32:56 2023

@author: zameer
"""
#import th file----------------
# MPG VS HP
import pandas as pd
df = pd.read_csv("Cars_4vars.csv")
df
df.shape
df.head()

#EDA-------------------------
import matplotlib.pyplot as plt
plt.scatter(df['MPG'],df['HP'])
plt.show()
df.corr()

Y = df["MPG"]
X = df[["HP"]]

#model fitting-----------------------
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)

#metrics-------------------------
import numpy as np
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("mean squared error",mse.round(2))
print("root mean square error",np.sqrt(mse).round(2))

#===========================================

# MPG VS VOL
import matplotlib.pyplot as plt
plt.scatter(df['MPG'],df['VOL'])
plt.show()
df.corr()

Y = df["MPG"]
X = df[["VOL","HP"]]

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)

import numpy as np
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("mean squared error",mse.round(2))
print("root mean square error",np.sqrt(mse).round(2))

# MPG VS SP
import matplotlib.pyplot as plt
plt.scatter(df['MPG'],df['SP'])
plt.show()
df.corr()

Y = df["MPG"]
X = df[["SP","VOL","HP"]]

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)

import numpy as np
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("mean squared error",mse.round(2))
print("root mean square error",np.sqrt(mse).round(2))

# MPG VS WT

import matplotlib.pyplot as plt
plt.scatter(df['MPG'],df['WT'])
plt.show()
df.corr()

Y = df["MPG"]
X = df[["WT","VOL","SP","HP"]]

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)

import numpy as np
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("mean squared error",mse.round(2))
print("root mean square error",np.sqrt(mse).round(2))


 




