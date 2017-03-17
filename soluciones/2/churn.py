import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
 
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE

cmap_bold = colors.ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


# 3. Load red wine data.
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

print X_scaled.shape
print y

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

reducer = Isomap(3, n_components=2)
X_reduced = reducer.fit_transform(X)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=cmap_bold)

plt.show()
raw_input()
 
