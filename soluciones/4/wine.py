import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.colors as colors

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

#cmap_bold = colors.ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

data = pd.read_csv('data/winequality-red.csv', sep=';')

y = data.quality
X = data.drop('quality', axis=1)

print data.head()
print data.describe()

X_train, X_test, y_train, y_test = train_test_split(X,y)

#scaler = preprocessing.StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

clf = RandomForestClassifier()
clf.fit(X_train,y_train)

print "RandomForestClassifier test score:", accuracy_score(y_test, clf.predict(X_test), normalize=True)
print  confusion_matrix(y_test, clf.predict(X_test))
 
best_score = 0.0

for C in [0.1,1,10,100,1000]:
    for tol in [0.00001, 0.00001, 0.0001, 0.001, 0.01]:
        clf = LinearSVC(C=C, tol=tol)
        score = cross_val_score(clf, X_train, y_train, cv=5).mean()
        if best_score < score:
            best_score = score
            print "Best SVM valid score:", score
            best_params = dict(C=C, tol=tol)

clf = LinearSVC(**best_params)
clf.fit(X_train,y_train)

print "SVM test score:", accuracy_score(y_test, clf.predict(X_test), normalize=True)
print  confusion_matrix(y_test, clf.predict(X_test))
