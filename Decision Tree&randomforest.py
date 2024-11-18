



import pandas as pd
df = pd.read_csv("sales.csv")
df.head()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

df['ShelveLoc'] = LE.fit_transform(df['ShelveLoc'])
df['Urban'] = LE.fit_transform(df['Urban'])
df["US"] = LE.fit_transform(df['US'])
df["high"] = LE.fit_transform(df['high'])

Y = df["high"]
X = df.iloc[:,1:11]
#list(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test  = train_test_split(X,Y)

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini')

DT.fit(X_train,Y_train)
Y_pred_train = DT.predict(X_train)
Y_pred_test = DT.predict(X_test)

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,Y_pred_train)
ac2 = accuracy_score(Y_test,Y_pred_test)
print("Training Accuracy", ac1.round(3))
print("Test Accuracy", ac2.round(3))

print("Total number of nodes",DT.tree_.node_count) # counting the number of nodes
print("Total maximum depth of tree",DT.tree_.max_depth) # number of levels


#====================================================================
# cross validation
#====================================================================

DT = DecisionTreeClassifier(criterion='gini',max_depth=6)

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    DT.fit(X_train,Y_train)
    Y_pred_train = DT.predict(X_train)
    Y_pred_test  = DT.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))


k1 = pd.DataFrame(training_accuracy)
k2 = pd.DataFrame(test_accuracy)

print("Cross validation training results:",k1.mean().round(2))
print("Cross validation test results:",k2.mean().round(2))
print('variance between train and test ',(k1.mean()-k2.mean()).round(2))

#==================================================================================
#Bagging Classifier

from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                        n_estimators=100,
                        max_samples=0.6,
                        max_features=0.7)

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    bag.fit(X_train,Y_train)
    Y_pred_train = bag.predict(X_train)
    Y_pred_test  = bag.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))


k1 = pd.DataFrame(training_accuracy)
k2 = pd.DataFrame(test_accuracy)

print("Cross validation training results:",k1.mean().round(2))
print("Cross validation test results:",k2.mean().round(2))
print('variance between train and test ',(k1.mean()-k2.mean()).round(2))

#=================================================================
# RandomForest Classifier

from sklearn.ensemble import RandomForestClassifier 
RFC = RandomForestClassifier(n_estimators=100,max_depth=5,
                        max_samples=0.6,
                        max_features=0.7)

training_accuracy = []
test_accuracy = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    RFC.fit(X_train,Y_train)
    Y_pred_train = RFC.predict(X_train)
    Y_pred_test  = RFC.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))


k1 = pd.DataFrame(training_accuracy)
k2 = pd.DataFrame(test_accuracy)

print("Cross validation training results:",k1.mean().round(2))
print("Cross validation test results:",k2.mean().round(2))
print('variance between train and test ',(k1.mean()-k2.mean()).round(2))





























