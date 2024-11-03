# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:16:04 2023

@author: excel
"""

# step1: import the file
import numpy as np
import pandas as pd
df = pd.read_csv("Cars_4vars.csv")
df.shape
df.head()

# statsmodel
# model1
import statsmodels.formula.api as smf
model = smf.ols('MPG ~ HP+VOL',data=df).fit()
model.fittedvalues # y_predicted value
model.resid  # residuals  --> y - ypred
mse = np.mean(model.resid ** 2)
print("Mean square Error", mse)
r2 = model.rsquared
print("R square", r2)

# model2
model = smf.ols('MPG ~ HP+WT',data=df).fit()
model.fittedvalues # y_predicted value
model.resid  # residuals  --> y - ypred
mse = np.mean(model.resid ** 2)
print("Mean square Error", mse)
r2 = model.rsquared
print("R square", r2)

#=============================================
#calculating the vif for above two models
import statsmodels.formula.api as smf
model = smf.ols('HP ~ VOL',data=df).fit()
r2 = model.rsquared
print("R square", r2)
VIF = 1 / (1-r2)
print("VIF:", VIF)

model = smf.ols('VOL ~ HP',data=df).fit()
r2 = model.rsquared
print("R square", r2)
VIF = 1 / (1-r2)
print("VIF:", VIF)

# SP AND HP
model = smf.ols('HP ~ SP',data=df).fit()
r2 = model.rsquared
print("R square", r2)
VIF = 1 / (1-r2)
print("VIF:", VIF)










