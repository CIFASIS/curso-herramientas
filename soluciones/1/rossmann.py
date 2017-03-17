# -*- coding: utf-8 -*-
# ----

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----

data = pd.read_csv('data/rossmann.csv.zip')

# ----

print data.head()
print data.describe()
print data.dtypes

def count_unique(column):
    return len(column.unique())

print data.apply(count_unique, axis=0)
print data.isnull().any()

# ----

store_data = data[data.Store==150]
store_data = store_data[map(lambda x: "2013" in x, store_data.Date)].sort('Date')

#También funcionaría:
#store_data = data[data.Sotre==150].sort('Date')

print store_data.head()
print store_data.tail()

plt.plot(store_data.Sales[:365])
plt.show()

# ----

plt.scatter(x=store_data[data.Open==1].Promo, y=store_data[data.Open==1].Sales, alpha=0.1)
plt.xlabel('Promo')
plt.ylabel('Sales')

plt.show()

# ----

print store_data[(store_data.Open == 1) & (store_data.Promo == 1)].Sales.mean()
print store_data[(store_data.Open == 1) & (store_data.Promo == 0)].Sales.mean()
