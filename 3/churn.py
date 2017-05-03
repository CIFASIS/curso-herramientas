# -*- coding: utf-8 -*-
# ----

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

data = pd.read_csv('data/churn.csv')

y = data['Churn?'] == "True."

to_drop = ['State','Area Code','Phone','Churn?']
X = data.drop(to_drop,axis=1)

yes_no_cols = ["Int'l Plan","VMail Plan"]
X[yes_no_cols] = X[yes_no_cols] == 'yes'

scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)
