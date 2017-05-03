# -*- coding: utf-8 -*-
# ----

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, TSNE

data = pd.read_csv('data/churn.csv')

print data.head()
print data.describe()

y = data['Churn?'] == "True."

to_drop = ['State','Area Code','Phone','Churn?']
X = data.drop(to_drop,axis=1)

yes_no_cols = ["Int'l Plan","VMail Plan"]
X[yes_no_cols] = X[yes_no_cols] == 'yes'

print X.head()

# ----
# ejercicios
# ----
