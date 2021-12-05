# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 21:46:31 2021

@author: Dhruv
"""
import os 
import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics 
import numpy as np
#import xgboost xgb

features = pd.read_csv('data/features.csv')
df_firmness = pd.read_csv('data/firmness.csv')

df = df_firmness.merge(features, on='fid', how='left')

# =============================================================================
# x_data = features
# x_data = features.set_index('fid')
# y_data = df_firmness[['fid', 'firmness']]
# y_data = y_data.set_index('fid')
# 
# =============================================================================

train_x, test_x, train_y, test_y = train_test_split(df.iloc[:,5:df.shape[1]], df[['firmness']], test_size=0.2, random_state=42)
train_x[:5]
train_y[:5]
train_x.shape
test_x.shape

lr = LinearRegression()
lr.fit(train_x, train_y)

pred_y = lr.predict(test_x)
pred_y[:5]
test_y[:5]

rmse = np.sqrt(metrics.mean_squared_error(test_y, pred_y))

pred_x = lr.predict(train_x)

rmsetrain = np.sqrt(metrics.mean_squared_error(train_y, pred_x))
print(rmsetrain)

test_y['pred_y'] = pred_y
























