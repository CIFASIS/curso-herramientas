import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

cmap_bold = colors.ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
data = pd.read_csv('data/churn.csv')

y = data['Churn?'] == "True."

to_drop = ['State','Area Code','Phone','Churn?']
X = data.drop(to_drop,axis=1)

yes_no_cols = ["Int'l Plan","VMail Plan"]
X[yes_no_cols] = X[yes_no_cols] == 'yes'

scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

reducer = Isomap(3, n_components=2)
X_reduced = reducer.fit_transform(X_scaled)

#plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=cmap_bold)
#plt.show()

X = X_reduced

est = KMeans(n_clusters=3)
est.fit(X)
labels = est.labels_

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap=cmap_bold)
plt.show()

X = data.drop(to_drop,axis=1)
yes_no_cols = ["Int'l Plan","VMail Plan"]
X[yes_no_cols] = X[yes_no_cols] == 'yes'

X_train, X_test, y_train, y_test = train_test_split(X,labels)

clf = RandomForestClassifier()
clf.fit(X_train,y_train)

print "RandomForestClassifier score:", accuracy_score(y_test, clf.predict(X_test), normalize=True)
print  confusion_matrix(y_test, clf.predict(X_test))

print clf.feature_importances_ 
important_columns = X.columns[clf.feature_importances_>0.1].values
important_columns = np.append(important_columns, ['Churn?'])

#print important_columns

to_drop = ['State','Area Code','Phone']
X = data.drop(to_drop,axis=1)

total = float(X['Churn?'].count())

print "Proporciones totales"
print X[X['Churn?'] == "True."]['Churn?'].count() / total
print X[X['Churn?'] == "False."]['Churn?'].count() / total

cluster0 = X.loc[labels==0,important_columns]

total = float(cluster0['Churn?'].count())

print cluster0.head()

print "Proporciones cluster rojo"
print cluster0[cluster0['Churn?'] == "True."]['Churn?'].count() / total
print cluster0[cluster0['Churn?'] == "False."]['Churn?'].count() / total

cluster1 = X.loc[labels==1,important_columns] 

total = float(cluster1['Churn?'].count())

print cluster1.head()

print "Proporciones cluster verde"
print cluster1[cluster1['Churn?'] == "True."]['Churn?'].count() / total
print cluster1[cluster1['Churn?'] == "False."]['Churn?'].count() / total

cluster2 = X.loc[labels==2,important_columns] 

total = float(cluster2['Churn?'].count())

print cluster2.head()

print "Proporciones cluster azul"
print cluster2[cluster2['Churn?'] == "True."]['Churn?'].count() / total
print cluster2[cluster2['Churn?'] == "False."]['Churn?'].count() / total
