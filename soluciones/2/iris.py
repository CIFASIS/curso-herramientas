import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()
X = iris.data
y = iris.target

reducer = PCA(n_components=3)
X_reduced = reducer.fit_transform(X)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
plt.show()

plt.scatter(X_reduced[:, 1], X_reduced[:, 2], c=y)
plt.show()
