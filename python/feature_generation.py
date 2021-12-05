# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:35:21 2021

@author: Dhruv
"""

import argparse
import os
import torch
import numpy as np
import cv2
import hdf5storage as hdf5
import sys

from scipy.io import loadmat
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

hs_path = './Documents/Hyperspectraldata/hsdata' #specify the path of the directory that has all the hyperspectral mat files
file_list_1 = []
fileCount = 0
for filename in os.listdir(hs_path):
    fileCount += 1
    split_list = []
    spectral = hdf5.loadmat(hs_path + "/" + filename)
    #removing comments
    #img = spectral['cube']  # 31 channels (482,512,31)
    feature = []
    hsMean = []
    hsStd = []
    hsMax = []
    hsMin = []
    for i in range(31):
      img = spectral['cube'][:, :, i]
      hsMean.append(np.mean(img))
      hsStd.append(np.std(img))
      hsMax.append(np.max(img))
      hsMin.append(np.min(img))
      feature = pd.DataFrame(hsMean)
      feature.reset_index(inplace=True)
      feature.columns = ['id', 'mean']
      feature['id'] = feature['id'].apply(str)
      feature['std'] = hsStd
      feature['max'] = hsMax
      feature['min'] = hsMin
      feature['fid'] = filename.partition('.')[0]
    if fileCount == 1:
      feature_wide = feature.pivot_table(index=["fid"], 
                      columns='id', 
                      values=['mean', 'std', 'max', 'min'])
      feature_wide.columns = [''.join(col) for col in feature_wide.columns]
      feature_wide.reset_index(inplace=True)
    else:
      feature_wide_temp = feature.pivot_table(index=["fid"], 
                      columns='id', 
                      values=['mean', 'std', 'max', 'min'])
      feature_wide_temp.columns = [''.join(col) for col in feature_wide_temp.columns]
      feature_wide_temp.reset_index(inplace=True)
      feature_wide = pd.concat([feature_wide,feature_wide_temp])

feature_wide.to_csv(hs_path + 'features.csv')
