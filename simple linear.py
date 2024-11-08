# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 11:07:33 2024

@author: zameer
"""

import numpy as np
age = np.array([[21],[23],[27],[32],[34],[37]])
weight = np.array([68,74,74,89,79,84])

import matplotlib.pyplot as plt
plt.scatter(age,weight)
plt.show()

from sklearn.linear_model import LinearRegression
LR = LinearRegression()

# X   Y
LR.fit(age,weight)

# Bias ---> Bo
LR.intercept_

# coeffient ---> B1
LR.coef_


# Bo+B1X1

49.57425742574257+0.98019802*24
49.57425742574257+0.98019802*30

