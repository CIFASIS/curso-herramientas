# -*- coding: utf-8 -*-
# ----

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# ----
# ejercicios
# ----

from sklearn.model_selection import cross_val_score

data = pd.read_csv('data/titanic.csv')

# ----

#print data.shape
#print data.head()
# sumario de columnas numéricas
#print data.describe()
#print data.isnull().any()

# ----
# salida
# ----

#print data[data['Age'].isnull()].count()
#print data[data['Age'].isnull()].head()
data['Age'].fillna(data['Age'].median(), inplace=True)
#print data.describe()

data.drop(['PassengerId','Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
data = pd.get_dummies(data)
data = pd.get_dummies(data, columns=['Pclass'])

X,y = data.drop(['Survived'], axis=1), data['Survived']

print X

clf = RandomForestClassifier()
score = cross_val_score(clf, X, y, cv=5)

print "RandomForestClassifier score:", score.mean()

for C in [0.1,1,10,100,1000]:
    for tol in [0.00001, 0.00001, 0.0001, 0.001, 0.01]:
        clf = SVC(kernel="linear", C=C, tol=tol)
        score = cross_val_score(clf, X, y, cv=5)
        print "SVM score:", score.mean()
