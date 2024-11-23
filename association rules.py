"""
Created on Thu Nov 23 15:17:09 2023
"""

pip install apyori

import pandas as pd

df = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
df.shape
df

# l1 = [[A, B] ,  [A, C, D] , [A, B,C,D]]

trans = []

for i in range(0, 7501):
  trans.append([str(df.values[i,j]) for j in range(0, 20)])
trans


from apyori import apriori
rules = apriori(transactions = trans, 
                min_support = 0.003, 
                min_confidence = 0.2, 
                min_lift = 3, 
                min_length = 2, 
                max_length = 2)

rules

results = list(rules)
results
len(results)


results[0]


a = []
b = []
c = []
d = []
e = []

for i in range(0,9):
    a.append(results[i][1]) # support
    b.append(results[i][2][0][0]) #base itme
    c.append(results[i][2][0][1]) # add itme
    d.append(results[i][2][0][2]) # confidence
    e.append(results[i][2][0][3]) # lift

df_new = pd.concat([pd.DataFrame(a),
                    pd.DataFrame(b),
                    pd.DataFrame(c),
                    pd.DataFrame(d),
                    pd.DataFrame(e)],axis=1)
          
k1 = ['Support','Baseitem','AddItem','Confidence','Lift']

df_new.columns = k1

df_new











