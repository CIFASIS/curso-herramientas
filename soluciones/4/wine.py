import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

cmap_bold = colors.ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

data = pd.read_csv('data/winequality-red.csv', sep=';')

y = data.quality
X = data.drop('quality', axis=1)

print data.head()
print data.describe()

scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled,y)

clf = RandomForestClassifier()
clf.fit(X_train,y_train)

print "RandomForestClassifier score:", accuracy_score(y_test, clf.predict(X_test), normalize=True)
print  confusion_matrix(y_test, clf.predict(X_test))
 
for C in [0.1,1,10,100,1000]:
    for tol in [0.00001, 0.00001, 0.0001, 0.001, 0.01]:
        clf = SVC(kernel="linear", C=C, tol=tol)
        score = cross_val_score(clf, X_train, y_train, cv=5)
        print "SVM score:", score.mean()
