# -*- coding: utf-8 -*-
# ----

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, TSNE

data = pd.read_csv('data/winequality-red.csv', sep=';')

y = data.quality
X = data.drop('quality', axis=1)

print data.head()
print data.describe()

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

print "media:", X.mean(axis=0)
print "std dev:", X.std(axis=0)

# PCA

reducer = PCA(n_components=2)
X_reduced = reducer.fit_transform(X)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
plt.show()

reducer = KernelPCA(n_components=2)
X_reduced = reducer.fit_transform(X)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
plt.show()

for gamma in [0.0001, 0.001 , 0.01, 0.1, 0.2, 0.5, 0.7, 0.9]:
    print "KernelPCA con gamma:", gamma
    reducer = KernelPCA(n_components=2,  kernel="rbf", gamma=gamma)
    X_reduced = reducer.fit_transform(X)

    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
    plt.show()


for pers in [5, 7, 10, 20, 25, 30, 50]:
    print "T-SNE con perprexity:", pers
    reducer = TSNE(n_components=2,  perplexity=pers)
    X_reduced = reducer.fit_transform(X)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
    plt.show()

for n_neighbors in [2, 3, 4, 5, 7, 10]:
    print "Isomap con número de vecinos:", n_neighbors
    reducer = Isomap(n_components=2, n_neighbors=n_neighbors)
    X_reduced = reducer.fit_transform(X)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
    plt.show()
