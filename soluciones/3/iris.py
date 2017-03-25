import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target


for k in range(2,10):

    est = KMeans(n_clusters=k)
    est.fit(X)
    labels = est.labels_
    print k, silhouette_score(X, labels)

    plt.scatter(X[:,0], X[:,1], c = labels)
    plt.show()

    plt.scatter(X[:,0], X[:,2], c = labels)
    plt.show()


for k in range(2,10):

    est = AgglomerativeClustering(n_clusters=k)
    est.fit(X)
    labels = est.labels_
    print k, silhouette_score(X, labels)

    plt.scatter(X[:,0], X[:,1], c = labels)
    plt.show()

    plt.scatter(X[:,0], X[:,2], c = labels)
    plt.show()
