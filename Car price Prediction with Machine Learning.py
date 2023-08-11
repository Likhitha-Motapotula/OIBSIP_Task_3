#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from statsmodels.formula.api import ols


# In[2]:


df = pd.read_csv("F:/data set/CarPrice_Assignment.csv")


# In[3]:


df.head()


# In[5]:


df.dtypes


# In[6]:


df.isna().sum()


# In[7]:


df.corr()


# In[8]:


correlation_matrix = df[['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize', 'boreratio', 'horsepower', 'citympg', 'highwaympg']].corr()
correlation_matrix


# In[9]:


sns.boxplot(x = 'fueltype', y = 'price', data = df)


# In[10]:


sns.boxplot(x = 'aspiration', y = 'price', data = df)


# In[11]:


sns.boxplot(x = 'doornumber', y = 'price', data = df)


# In[12]:


sns.boxplot(x = 'carbody', y = 'price', data = df)


# In[13]:


sns.boxplot(x = 'drivewheel', y = 'price', data = df)


# In[14]:


sns.boxplot(x = 'enginelocation', y = 'price', data = df)


# In[15]:


sns.boxplot(x = 'enginetype', y = 'price', data = df)


# In[16]:


sns.boxplot(x = 'cylindernumber', y = 'price', data = df)


# In[17]:


sns.boxplot(x = 'fuelsystem', y = 'price', data = df)


# In[18]:


df_subset = df[['wheelbase','carlength','carwidth', 'curbweight', 'enginesize', 'boreratio','horsepower', 'citympg', 'highwaympg', 'enginelocation', 'drivewheel', 'price']]


# In[19]:


df1 = df_subset[['wheelbase','carlength','carwidth', 'curbweight', 'enginesize', 'boreratio','horsepower', 'citympg', 'highwaympg']]


# In[20]:


scaler = StandardScaler()
scaler


# In[24]:


df_scaled = scaler.fit_transform(df1)
df_scaled = pd.DataFrame(df_scaled, columns = df1.columns)


# In[25]:


df_scaled.head()


# In[26]:


df_concat = pd.concat([df_scaled, df_subset[['enginelocation', 'drivewheel', 'price']]], axis = 1)


# In[27]:


df_concat.head()


# In[28]:


dummy_df = pd.get_dummies(df_concat, columns=['enginelocation'], drop_first=True)
dummy_df = pd.get_dummies(dummy_df, columns = ['drivewheel'], drop_first = True)
dummy_df


# In[29]:


X = dummy_df.drop(columns = ['price'])
Y = dummy_df['price']


# In[30]:


xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.25, random_state = 42)


# In[31]:


xtrain.shape, xtest.shape, ytrain.shape, ytest.shape


# In[32]:


lm = LinearRegression()
lm


# In[33]:


lm.fit(xtrain, ytrain)


# In[34]:


lm.intercept_


# In[35]:


lm.coef_


# In[36]:


y_hat = lm.predict(xtrain)


# In[37]:


ax = sns.distplot(ytrain, hist = False, color = 'r', label = 'actual values')
sns.distplot(y_hat, hist = False, color = 'b', label = 'fitted_values', ax = ax)

plt.show()


# In[38]:


y_hat1 = lm.predict(xtest)


# In[39]:


lm.score(xtest, ytest)


# In[40]:


ols_formula = "price ~ wheelbase + carlength + carwidth + curbweight + enginesize + boreratio + horsepower + citympg + highwaympg + C(enginelocation) + C(drivewheel)"


# In[41]:


OLS = ols(formula = ols_formula, data = df_concat)


# In[42]:


model = OLS.fit()


# In[43]:


model.summary()


# In[ ]:




