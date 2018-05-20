import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://raw.githubusercontent.com/Geoyi/Cleaning-Titanic-Data/master/titanic_original.csv')
print('=========================================')
print('DATA:')
print(df.head(6))
print(df.info())

print(df.info())

is_male = pd.get_dummies(df['sex'],drop_first=True)
df = pd.concat([df,is_male],axis = 1)

print('dorp not numeric columns:')
df = df.drop(['sex','name','cabin','embarked','boat','body','home.dest','ticket'],axis = 1)
print(df.info())

print('drop NaN values:')
df = df.dropna()
print(df.info())

print('CLASSIFICATION:')


X = df.drop('survived',axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state=4)

print('classifier: decisionTree')
dt_cls = DecisionTreeClassifier(random_state=4)
dt_cls.fit(X_train,y_train)
predictions = dt_cls.predict(X_test)

print('accuracy on test sample(30%):')
print(accuracy_score(y_test,predictions))
