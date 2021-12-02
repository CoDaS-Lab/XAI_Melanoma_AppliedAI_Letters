# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 15:02:36 2021

@author: Harshit Bokadia
"""
import torch
import torch.nn.functional as F

from PIL import Image
from CaptumVisual import visualize_image_attr_new

import os
import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torchvision
from torchvision import models
from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz


import matplotlib
from scipy.linalg import eigh 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as gaussian
from torchvision import datasets, transforms
import cv2
import csv
import math
import random
# import imutils
import torchvision.utils as utils
import argparse
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as torch_transforms
from networks_new import AttnVGG, VGG
from loss import FocalLoss
#from data import preprocess_data_2016, preprocess_data_2017, ISIC
from data import *
from utilities import *
from transforms import *

from skimage import data, filters, color, morphology
from skimage.segmentation import flood, flood_fill
import math

data_train = torch.load('features_train_final.pt', map_location ='cpu')
data_test = torch.load('features_test_final.pt', map_location ='cpu')

#######training data
x_train_A_OG = np.array(data_train['Asymmetry_Scores'], dtype=object)
x_train_B_OG = np.array(data_train['Border_Scores'], dtype=object)
x_train_C_OG = np.array(data_train['Color_Scores'], dtype=object)

empty_index_A = [i for i,x in enumerate(x_train_A_OG) if x == []]
empty_index_B = [j for j,y in enumerate(x_train_B_OG) if y == []]

UQ = list(set(empty_index_A + empty_index_B))


x_train_A = np.delete(x_train_A_OG, UQ, 0)
x_train_B = np.delete(x_train_B_OG, UQ, 0)
x_train_C = np.delete(x_train_C_OG, UQ, 0)

x_train = np.column_stack((x_train_A, x_train_B, x_train_C))

x_train_A_OG[UQ]= np.nan 
x_train_B_OG[UQ]= np.nan 


x_train_NAN = np.column_stack((x_train_A_OG, x_train_B_OG, x_train_C_OG))

#########test data
x_test_A = np.array(data_test['Asymmetry_Scores'], dtype=object)
x_test_B = np.array(data_test['Border_Scores'], dtype=object)
x_test_C = np.array(data_test['Color_Scores'], dtype=object)

empty_index_A = [i for i,x in enumerate(x_test_A) if x == []]
empty_index_B = [j for j,y in enumerate(x_test_B) if y == []]

UQ_test = list(set(empty_index_A + empty_index_B))

x_test_A[UQ_test]= np.nan 
x_test_B[UQ_test]= np.nan 


x_test = np.column_stack((x_test_A, x_test_B, x_test_C))


y_train = np.array(data_train['ground_truth'], dtype='f') 
y_test = np.array(data_test['ground_truth'], dtype='f') 
y_train =  np.delete(y_train, UQ, 0)


from sklearn.linear_model import LogisticRegression

# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression(class_weight = 'balanced', solver ='lbfgs', max_iter=200)

logisticRegr.fit(x_train, y_train)

          
TFM_predictions_train = np.zeros([x_train_NAN.shape[0], 1])
TFM_probs_train = np.zeros([x_train_NAN.shape[0], 2])

for i in range(x_train_NAN.shape[0]):
      if np.isnan(x_train_NAN[i, 0]):
          TFM_predictions_train[i] = np.nan
          TFM_probs_train[i] = np.nan
      else:
          TFM_predictions_train[i] = logisticRegr.predict(x_train_NAN[i].reshape([1, 3]))
          TFM_probs_train[i] = logisticRegr.predict_proba(x_train_NAN[i].reshape([1, 3])) 
          
          
TFM_predictions_test = np.zeros([x_test.shape[0], 1])
TFM_probs_test = np.zeros([x_test.shape[0], 2])

for i in range(x_test.shape[0]):
      if np.isnan(x_test[i, 0]):
          TFM_predictions_test[i] = np.nan
          TFM_probs_test[i] = np.nan
      else:
          TFM_predictions_test[i] = logisticRegr.predict(x_test[i].reshape([1, 3]))
          TFM_probs_test[i] = logisticRegr.predict_proba(x_test[i].reshape([1, 3]))            
     
# Use score method to get accuracy of model
score = logisticRegr.score(x_train, y_train)
print(score)

import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(y_train, predictions)
print(cm)

plt.figure(figsize=(9,9))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix', size = 15)
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["0", "1"], rotation=45, size = 10)
plt.yticks(tick_marks, ["0", "1"], size = 10)
plt.tight_layout()
plt.ylabel('Actual label', size = 15)
plt.xlabel('Predicted label', size = 15)



coeff = logisticRegr.coef_
print(coeff)

int_cpt =  logisticRegr.intercept_
print(int_cpt)

# np.save('x_train.npy', x_train)
# np.save('x_test.npy', x_test)
# np.save('y_train.npy', y_train)
# np.save('y_test.npy', y_test)

torch.save({
        'TFM_preds_train': TFM_predictions_train,
        'TFM_preds_test': TFM_predictions_test,
        'TFM_probs_train': TFM_probs_train,
        'TFM_probs_test': TFM_probs_test    
      }, 'TFM_final_new.pt')

# data_TFM = torch.load('TFM_final.pt', map_location ='cpu')
