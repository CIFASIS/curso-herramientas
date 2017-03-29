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

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=cmap_bold)
plt.show()

X = X_reduced

best_labels = []
best_score = -1

for k in range(2,10):

    est = KMeans(n_clusters=k)
    est.fit(X)
    labels = est.labels_
    score = silhouette_score(X, labels)
    print k,score

    if best_score < score:
        best_score = score
        best_labels = list(labels)

print "mejor score con kmeans", best_score

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=best_labels, cmap=cmap_bold)
plt.show()


for k in range(2,10):

    est = AgglomerativeClustering(n_clusters=k)
    est.fit(X)
    labels = est.labels_
    score = silhouette_score(X, labels)
    print k,score

    if best_score < score:
        best_score = score
        best_labels = list(labels)

print "mejor score con hierarchical clustering", best_score

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=best_labels, cmap=cmap_bold)
plt.show()
