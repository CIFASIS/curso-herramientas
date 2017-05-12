# -*- coding: utf-8 -*-
# ----

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# ----

data = pd.read_csv('data/rossmann.csv.zip')

# ----

print data.head()
print data.describe()
print data.dtypes

lr = LinearRegression()

#data['Date'] = pd.to_datetime(data['Date'])    
#data['Date'] = (data['Date'] - data['Date'].min())  / np.timedelta64(1,'D')

store_data = data[data.Store==150].sort('Date')

X_store = pd.get_dummies(data[data.Store!=150], columns=['DayOfWeek', 'StateHoliday']).drop(['Sales', 'Store', 'Date', 'Customers'], axis=1).values
y_store = pd.get_dummies(data[data.Store!=150], columns=['DayOfWeek', 'StateHoliday']).Sales.values

lr.fit(X_store, y_store)

y_store_predict = lr.predict(pd.get_dummies(store_data, columns=['DayOfWeek', 'StateHoliday']).drop(['Sales', 'Store', 'Date', 'Customers'], axis=1).values)

plt.plot(store_data.Sales[:365].values, label="ground truth")
plt.plot(y_store_predict[:365], c='r', label="prediction")
plt.legend()
plt.show()
