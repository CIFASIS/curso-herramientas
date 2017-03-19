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
from sklearn import datasets

iris = datasets.load_iris()
X,y = iris.data, iris.target

clf = RandomForestClassifier()
score = cross_val_score(clf, iris.data, iris.target, cv=5)

print "RandomForestClassifier score:", score.mean()

for C in [0.1,1,10,100,1000]:
    for tol in [0.00001, 0.00001, 0.0001, 0.001, 0.01]:
        clf = SVC(kernel="linear", C=C, tol=tol)
        score = cross_val_score(clf, iris.data, iris.target, cv=5)
        print "SVM score:", score.mean()

