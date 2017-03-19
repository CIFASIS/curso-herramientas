# -*- coding: utf-8 -*-
# ----

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data/titanic.csv')
data_notnull = pd.read_csv('data/titanic.csv')

# ----

print data.shape
print data.head()
# sumario de columnas numéricas
print data.describe()
print data.isnull().any()

# ----
# salida
# ----

print data[data['Age'].isnull()].count()
print data[data['Age'].isnull()].head()
data['Age'].fillna(data['Age'].median(), inplace=True)
print data.describe()

# ----
# salida
# ----

survived_sex = data[data['Survived']==1]['Sex'].value_counts()
dead_sex = data[data['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(15,8))

plt.show()

# ----
# gráfico
# ----

figure = plt.figure(figsize=(15,8))
plt.hist([data[data['Survived']==1]['Age'],data[data['Survived']==0]['Age']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()

plt.show()

# ----
# gráfico
# ----
# ejercicio 1.1
# ----

data_notnull = data_notnull[data_notnull['Age'].notnull()]
figure = plt.figure(figsize=(15,8))
plt.hist([data_notnull[data_notnull['Survived']==1]['Age'],data_notnull[data_notnull['Survived']==0]['Age']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()

plt.show()

# ----


figure = plt.figure(figsize=(13,8))
plt.hist([data[data['Survived']==1]['Fare'],data[data['Survived']==0]['Fare']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()

plt.show()

# ----
# ejercicio 1.2
# ----

survived_embarked = data[data['Survived']==1]['Embarked'].value_counts()
dead_embarked = data[data['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embarked,dead_embarked])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(15,8), color = ['g','r','y'])

plt.show()
