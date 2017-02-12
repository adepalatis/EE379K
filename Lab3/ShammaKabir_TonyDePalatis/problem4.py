# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 15:33:45 2017

@author: shammakabir
"""

import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib
import sklearn
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
#from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import cross_val_score



#config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook
#matplotlib inline
#train = pd.read_csv("/Users/shammakabir/Downloads/train.csv")
#test = pd.read_csv("/Users/shammakabir/Downloads/test.csv")
train = pd.read_csv("C:\Users\Tony\Downloads\\train.csv")
test = pd.read_csv("C:\Users\Tony\Downloads\\test.csv")
train.head()

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()

#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)

#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

model_ridge = Ridge(alpha=8.0,tol=0.012, solver='svd').fit(X_train, y)
predict = np.expm1(model_ridge.predict(X_test))

results = pd.DataFrame({"Id":test.Id, "SalePrice":predict})
results.to_csv("results.csv", index = False)

#RSME: 0.13029
#best i could get: 0.12312