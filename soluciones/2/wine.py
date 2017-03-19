import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, TSNE

data = pd.read_csv('data/winequality-red.csv', sep=';')

print data.head()
print data.describe()

y = data.quality
X = data.drop('quality', axis=1)

scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

print X_scaled.shape
print y

print X.mean(axis=0)
print X_scaled.mean(axis=0)

print X.std(axis=0)
print X_scaled.std(axis=0)

reducer = TSNE(n_components=10)
X_reduced = reducer.fit_transform(X_scaled)

print set(y)
#print(pca.explained_variance_)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)

plt.show()

reducer = PCA(n_components=10)
X_reduced = reducer.fit_transform(X_scaled)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)

plt.show()
