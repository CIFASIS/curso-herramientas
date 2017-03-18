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

print data.head()
print data.describe()

y = data['Churn?'] == "True."

to_drop = ['State','Area Code','Phone','Churn?']
X = data.drop(to_drop,axis=1)

yes_no_cols = ["Int'l Plan","VMail Plan"]
X[yes_no_cols] = X[yes_no_cols] == 'yes'

scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

#reducer = Isomap(3, n_components=2)
#X_reduced = reducer.fit_transform(X_scaled)

#plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=cmap_bold)
#plt.show()

#print X_scaled.shape
#print y

for k in range(2,10):

    est = AgglomerativeClustering(n_clusters=k)
    est.fit(X_scaled)
    labels = est.labels_
    print k, silhouette_score(X_scaled, labels)

    #plt.scatter(X[:,0], X[:,1], c = labels)
    #plt.show()

    #plt.scatter(X[:,0], X[:,2], c = labels)
    #plt.show()

assert(0)

print X.mean(axis=0)
print X_scaled.mean(axis=0)

print X.std(axis=0)
print X_scaled.std(axis=0)

reducer = PCA(n_components=10)
X_reduced = reducer.fit_transform(X)

print set(y)
#print(pca.explained_variance_)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=cmap_bold)

plt.show()

