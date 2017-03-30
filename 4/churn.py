# -*- coding: utf-8 -*-
# ----

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

# ----
# ejercicios
# ----

data = pd.read_csv('data/churn.csv')

y = data['Churn?'] == "True."

to_drop = ['State','Area Code','Phone','Churn?']
X = data.drop(to_drop,axis=1)

yes_no_cols = ["Int'l Plan","VMail Plan"]
X[yes_no_cols] = X[yes_no_cols] == 'yes'
