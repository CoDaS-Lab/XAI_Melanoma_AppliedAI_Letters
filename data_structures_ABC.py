# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 00:04:01 2022

@author: Harshit Bokadia

"""
# Color Threshold = 10
# Border threshold= 13

import torch
import torch.nn.functional as F

from PIL import Image


import os
import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torchvision
from torchvision import models
from torchvision import transforms

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

import torchvision.utils as utils
import argparse
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as torch_transforms
from networks_new import AttnVGG, VGG
from loss import FocalLoss
from data import *
from utilities import *
from transforms import *

from skimage import data, filters, color, morphology
from skimage.segmentation import flood, flood_fill
import math

np.random.seed(1110)
torch.manual_seed(1110)

import glob

rootdir =  'C:/Users\harsh\Desktop\HBK008\CoDaS Lab\Project XAI\data_2016\Train'
images_benign = [cv2.imread(file) for file in glob.glob(os.path.join(rootdir, 'benign', '*.jpg'))]
images_malignant = [cv2.imread(file) for file in glob.glob(os.path.join(rootdir, 'malignant', '*.jpg'))]

data_images = images_benign + images_malignant


rootdir_lesion =  'C:/Users\harsh\Desktop\HBK008\CoDaS Lab\Project XAI\data_2016\Train_Lesion'
images_benign_lesion = [cv2.imread(file) for file in glob.glob(os.path.join(rootdir_lesion, 'benign', '*.png'))]
images_malignant_lesion = [cv2.imread(file) for file in glob.glob(os.path.join(rootdir_lesion, 'malignant', '*.png'))]

data_images_lesion = images_benign_lesion + images_malignant_lesion

data = torch.load('extract_train_final_new.pt', map_location ='cpu')

# data_old = torch.load('features_test_final.pt', map_location ='cpu')

###############
dnn_probs = data['extract']

probs = []
for i in range(len(dnn_probs)):
    FF_temp = np.array(dnn_probs[i].cpu())
    
    probs.append(FF_temp)

probs = np.vstack(probs)

probs = torch.Tensor(probs)

prob_st = []

for i in range(len(probs)):
    probability = F.softmax(probs[i]).squeeze().cpu().detach().numpy()
    
    prob_st.append(probability)
    
prob_st = np.vstack(prob_st)

prob_st = torch.Tensor(prob_st)

prob_st = list(prob_st)
#########################


preds = data['predictions']

predictions = [item for sublist in preds for item in sublist]

Labels = data['Labels']



def variance(data, ddof=0):
     n = len(data)
     mean = sum(data) / n
     return sum((x - mean) ** 2 for x in data) / (n - ddof)



def stdev(data):
     var = variance(data)
     std_dev = math.sqrt(var)
     return std_dev
 
def plotMinMax(Xsub_rgb,labels=["R","G","B"]):
    print("______________________________")
    for i, lab in enumerate(labels):
        mi = np.min(Xsub_rgb[:,:,i])
        ma = np.max(Xsub_rgb[:,:,i])
        print("{} : MIN={:8.4f}, MAX={:8.4f}".format(lab,mi,ma))  
        
     
        
transformations = transforms.Compose([

    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),

]) 

##########################################################################        
        
def AsymmetryScore(mn, data_images):
    
    img = data_images_lesion[mn]
    # I = II[mn, :, :, :].unsqueeze(0)
    # I_extract = utils.make_grid(I, nrow=1, normalize=True, scale_each=True)
    # IM = I_extract
    # IM_Original = I_extract.numpy()
    # IM_Original = np.moveaxis(IM_Original, 0, -1)
    # img=IM_Original[:, :, :]
    # RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.figure()
    # plt.imshow(gray)
    # plt.axis('off')
    plt.imsave('OriginalImage.jpg',img)
        
    img = cv2.imread('OriginalImage.jpg')
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    contours,hierarchy = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    c = max(contours, key=cv2.contourArea)    
        
    # fit contour to ellipse and get ellipse center, minor and major diameters and angle in degree 
    ellipse = cv2.fitEllipse(c)
    (xc,yc),(d1,d2),angle = ellipse
    # print(xc,yc,d1,d1,angle)
    ELP = [xc, yc, d1, d2, angle]
    # draw ellipse
    result = img.copy()
    cv2.ellipse(result, ellipse, (0, 255, 0), 10)
    
    # draw circle at center
    xc, yc = ellipse[0]
    cv2.circle(result, (int(xc),int(yc)), 10, (100, 100, 100), -1)
    
    # plt.figure()
    # plt.imshow(result)
    
    
    # draw vertical line
    # compute major radius
    rmajor = max(d1,d2)/2
    if angle > 90:
        angle = angle - 90
    else:
        angle = angle + 90
    # print(angle)
    xtop = xc + math.cos(math.radians(angle))*rmajor
    ytop = yc + math.sin(math.radians(angle))*rmajor
    xbot = xc + math.cos(math.radians(angle+180))*rmajor
    ybot = yc + math.sin(math.radians(angle+180))*rmajor
    cv2.line(result, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (100, 100, 100), 15)

    
    angle_m = angle
    
    rminor = min(d1,d2)/2
    if angle_m > 90:
        angle_m = angle_m - 90
    else:
        angle_m = angle_m + 90
    # print(angle_m)
    
    xtop_m = xc + math.cos(math.radians(angle_m))*rminor
    ytop_m = yc + math.sin(math.radians(angle_m))*rminor
    xbot_m = xc + math.cos(math.radians(angle_m+180))*rminor
    ybot_m = yc + math.sin(math.radians(angle_m+180))*rminor
    cv2.line(result, (int(xtop_m),int(ytop_m)), (int(xbot_m),int(ybot_m)), (100, 100, 100), 15)
    
    cv2.imwrite("melanoma_ellipse.jpg", result)
    
    img_elfit = cv2.imread('melanoma_ellipse.jpg')
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(8,8))
    # plt.imshow(img_elfit)
    # plt.axis('off')
    # plt.title("Original Image with ellipse fit")
    # plt.show()
    
    # #############################################
    # #Rotation
    
    center = (xc, yc)
    
    img_asym = gray.copy()
    # print(img_asym.shape)
    
    height = img_asym.shape[0]
    width = img_asym.shape[1]
    
    # using cv2.getRotationMatrix2D() to get the rotation matrix
    
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
    # rotate the image using cv2.warpAffine
    rotated_image = cv2.warpAffine(src=img_asym, M=rotate_matrix, dsize=(width, height))
    
    cv2.imwrite('rotated_image.jpg', rotated_image)
    
    img_rt = cv2.imread('rotated_image.jpg')
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(8,8))
    # plt.imshow(img_rt)
    # plt.axis('off')
    # plt.title("Rotated Image")
    # plt.show()
    ##############################
    
    
    # # Cut the image in half
    # Minor axis
    from skimage import io, color
    img_as = img_rt.copy()
    img_as = color.rgb2gray(img_as)
    
    
    if 2*int(xc)<img_as.shape[1]:
        
        diff = img_as.shape[1]-(2*int(xc))
        padding_matrix = np.zeros([img_as.shape[0], int(diff)])
        s1 = img_as[:, :int(xc)]
        s1 = np.concatenate((padding_matrix, s1), axis=1)
        s2 = img_as[:, int(xc):img_as.shape[1]]
        
    else:

        diff = 2*int(xc)-img_as.shape[1]
        padding_matrix = np.zeros([img_as.shape[0], int(diff)])
        s1 = img_as[:, :int(xc)]
        s2 = img_as[:, int(xc):img_as.shape[1]]
        s2 = np.concatenate((s2, padding_matrix), axis=1)
    
    # plt.figure()
    # plt.imshow(s1)
    # plt.axis('off')
    
    # plt.figure()
    # plt.imshow(s2)
    # plt.axis('off')
    
    
    s1[s1 > 0] = 1 # Left
    s2[s2 > 0] = 10 # Right
    
    s7 = cv2.flip(s2, 1) # Right one flipped
    
    # plt.figure()
    # plt.imshow(s7)
    # plt.axis('off')
    # plt.title('Right Half flipped')
    
    
    overlap_new_hz = s1+s7

    overlap_new_hz[overlap_new_hz == 11] = 0 # overlap of both half
    overlap_new_hz[overlap_new_hz == 10] = 0 # pixels of right half
    overlap_new_hz[overlap_new_hz == 1] = 255 # pixels of left half

    # plt.figure()
    # plt.imshow(overlap_new_hz)
    # plt.axis('off')
    # plt.title('Pixels of assymetry on overlap in left half')
    
    
    s8 = cv2.flip(s1, 1)
    
    # plt.figure()
    # plt.imshow(s8)
    # plt.axis('off')
    # plt.title('Left Half flipped')
    
    
    overlap_new1_hz = s2+s8
    
    overlap_new1_hz[overlap_new1_hz == 11] = 0
    overlap_new1_hz[overlap_new1_hz == 1] = 0
    overlap_new1_hz[overlap_new1_hz == 10] = 255
    

    # plt.figure()
    # plt.imshow(overlap_new1_hz)
    # plt.axis('off')
    # plt.title('Pixels of assymetry on overlap in right half')
    
    
    assymetry_mask_hz = np.concatenate((overlap_new_hz, overlap_new1_hz), axis=1)
    
    if 2*int(xc)<img_as.shape[1]:
    
        # start = assymetry_mask_hz.shape[0]
        # end = assymetry_mask_hz.shape[1]
        assymetry_mask_final_hz = np.delete(assymetry_mask_hz, np.s_[0:diff], axis=1)
        
    else:
        
        # start = assymetry_mask_hz.shape[0]
        end = img_as.shape[1]
        assymetry_mask_final_hz = np.delete(assymetry_mask_hz, np.s_[end:(end+diff)], axis=1)
    
    # plt.figure()
    # plt.imshow(assymetry_mask_final_hz)
    # plt.axis('off')
    # plt.title("Asymmetry mask for minor axis")
    
##################################################################
# Major axis
    
    from skimage import io, color
    
    img_as = img_rt.copy()
    
    img_as = color.rgb2gray(img_as)
    
       
    if 2*int(yc)<img_as.shape[0]:
        
        diff = img_as.shape[0]-(2*int(yc))
        padding_matrix = np.zeros([int(diff), img_as.shape[1]])
        s3 = img_as[:int(yc), :]
        s3 = np.concatenate((padding_matrix, s3), axis=0)
        s4 = img_as[int(yc):img_as.shape[0], :]
        
    else:

        diff = 2*int(yc)-img_as.shape[0]
        padding_matrix = np.zeros([int(diff), img_as.shape[1]])
        s3 = img_as[:int(yc), :]
        s4 = img_as[int(yc):img_as.shape[0], :]
        s4 = np.concatenate((s4, padding_matrix), axis=0)    
    
    # plt.figure()
    # plt.imshow(s3)
    # plt.axis('off')
    # plt.title('Top Half')
    
    # plt.figure()
    # plt.imshow(s4)
    # plt.axis('off')
    # plt.title('Bottom Half')
    
    
    ss4 = s4.copy()
    s3[s3 > 0] = 1
    s4[s4 > 0] = 10
    
    s5 = cv2.flip(s4, 0)
    
    # plt.figure()
    # plt.imshow(s5)
    # plt.axis('off')
    # plt.title('Bottom Half flipped')
    
    overlap_new = s3+s5

    overlap_new[overlap_new == 11] = 0
    overlap_new[overlap_new == 10] = 0
    overlap_new[overlap_new == 1] = 255

    # plt.figure()
    # plt.imshow(overlap_new)
    # plt.axis('off')
    # plt.title('Pixels of assymetry on overlap in top half')
    
    
    # Overlap BOTTOM
    
    s6 = cv2.flip(s3, 0)
    
    # plt.figure()
    # plt.imshow(s6)
    # plt.axis('off')
    # plt.title('Top Half flipped')
    
    
    overlap_new1 = s4+s6
    
    overlap_new1[overlap_new1 == 11] = 0
    overlap_new1[overlap_new1 == 1] = 0
    overlap_new1[overlap_new1 == 10] = 255
    

    # plt.figure()
    # plt.imshow(overlap_new1)
    # plt.axis('off')
    # plt.title('Pixels of assymetry on overlap in bottom half')
    
    assymetry_mask = np.concatenate((overlap_new, overlap_new1), axis=0)
    
    if 2*int(yc)<img_as.shape[0]:

        assymetry_mask_final_vt = np.delete(assymetry_mask, np.s_[0:diff], axis=0)
        
    else:
        
        # start = assymetry_mask.shape[1]
        end1 = img_as.shape[0]
        assymetry_mask_final_vt = np.delete(assymetry_mask, np.s_[end1:(end1+diff)], axis=0)
    
    # plt.figure()
    # plt.imshow(assymetry_mask_final_vt)
    # plt.axis('off')
    # plt.title("Asymmetry mask for major axis")
    
    
    ##########################################################
    # add the two masks
    Asymmetry_FeatureMask = np.zeros((img_as.shape[0], img_as.shape[1]))
    Asymmetry_FeatureMask = assymetry_mask_final_hz+ assymetry_mask_final_vt
    
    
    # for i in range(int(Asymmetry_FeatureMask.shape[0])): 
    #     for j in range(int(Asymmetry_FeatureMask.shape[1])): 
    #         if Asymmetry_FeatureMask[i, j]> 0:
    #            Asymmetry_FeatureMask[i, j]=1
    #         else:
    #            Asymmetry_FeatureMask[i, j]=0 
    
    Asymmetry_FeatureMask[Asymmetry_FeatureMask > 0] = 1 
    # plt.figure()
    # plt.imshow(Asymmetry_FeatureMask)
    # plt.axis('off')
    # plt.title("Asymmetry Feature Mask")
    
    ##########################################################
    # Rotate back
    center = (xc, yc)
        
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=-angle, scale=1)
    # rotate the image using cv2.warpAffine
    rotated_image = cv2.warpAffine(src=Asymmetry_FeatureMask.copy(), M=rotate_matrix, dsize=(width, height))
    
    
    # for i in range(int(rotated_image.shape[0])): 
    #     for j in range(int(rotated_image.shape[1])): 
    #         if rotated_image[i, j]> 0:
    #            rotated_image[i, j]=1
    #         else:
    #            rotated_image[i, j]=0 
    
    rotated_image[rotated_image > 0] = 1 
    
   
    
    # sample = rotated_image.copy()
    
    # plt.imsave('OriginalImage.jpg',rotated_image)
        
    # img = cv2.imread('OriginalImage.jpg')
    
    # sample = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    # cv2.ellipse(sample, ellipse, (128, 128, 128), 20)
    # cv2.circle(sample, (int(xc),int(yc)), 10, (100, 100, 100), -1)
    # cv2.line(sample, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (100, 100, 100), 15)
    # cv2.line(sample, (int(xtop_m),int(ytop_m)), (int(xbot_m),int(ybot_m)), (100, 100, 100), 15)
    
    # # cv2.imwrite('rotated_image.jpg', sample)
    
    # # img__final = cv2.imread('rotated_image.jpg')
    
   
    # # plt.figure(figsize=(8,8))
    # #plt.imshow(rotated_image)
    # plt.figure()
    # plt.imshow(sample)
    # plt.axis('off')
    # #plt.title("ReRotated Asymmetry Feature Mask")
    # plt.show()
    
    
    
    
    ############################################################
    # Asymmetry score calculations
    
    pixels_nz = cv2.countNonZero(Asymmetry_FeatureMask) 

    lesion_area = cv2.countNonZero(gray.copy())
     
    Asymmetry_Score = (pixels_nz/ lesion_area) 
    
    # print('pixels', pixels)
    #print('Asymmetry FeatureScore', Asymmetry_Score)
    ratio = 0.8
    width = rotated_image.shape[0]
    height = rotated_image.shape[1]
    new_size = ratio * min(width, height)
    img = trF.center_crop(torch.tensor(rotated_image), new_size)
    PIL_image = Image.fromarray(np.uint8(img.numpy()))
    # PIL_image = Image.fromarray(img.numpy())
    # PIL_image = Image.fromarray(np.uint8(rotated_image))

    TS = transformations(PIL_image)
    
    IM = TS.numpy()
    IM = np.moveaxis(IM, 0, -1)
    cropped_AM=IM[:, :, :]
    
    ImageShape = [width, height]
    
    return Asymmetry_Score, cropped_AM, ELP, ImageShape

###########################################################################
 

def BorderScore(mn, data_images_lesion):
    
    img = data_images_lesion[mn]
    plt.imsave('OriginalImage.jpg',img)
        
    img = cv2.imread('OriginalImage.jpg')
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contours,hierarchy = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    c = max(contours, key=cv2.contourArea)
    #clist = c.tolist()
    
    # M = cv2.moments(gray)
    
    # cX = int(M["m10"] / M["m00"])
    # cY = int(M["m01"] / M["m00"])
    
    ellipse = cv2.fitEllipse(c)
    (xc,yc),(d1,d2),angle = ellipse
    #print(xc,yc,d1,d1,angle)
    
    cv2.circle(img, (int(xc), int(yc)), 5, (255, 255, 255), -1)

    # # cv2.putText(img, "C", (cX - 10, cY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # plt.imshow(img)
    # plt.axis('off')
    
    img_border = cv2.drawContours(img, c, -1, (0, 0, 255), 4)
    # # cv2.imshow('frame', img_border)
    
    # plt.imshow(img_border)
    # plt.axis('off')
    
    h, w, _ = img.shape
    contour = contours[0]


    CT=[]
    for j in range(int(c.shape[0])):
            ctemp= c[j, :, :]
            x= ctemp[0][0]
            y= ctemp[0][1]
            CTemp = [x, y]
            CT += CTemp
        
    CT = np.array(CT)
    CT = CT.reshape(c.shape[0],2)
    CTxmax = max(CT[:, 0])
    CTymax = max(CT[:, 1])

    divisions= 8
    # center = (cX,cY)
    center = (int(xc),int(yc))
    distance = 2*max(img.shape)
    # distance = 45
    CPie = {}
    sector_number = np.zeros(CT.shape[0])
    
    for i in range(divisions):
        # Get some start and end points that slice the image into pieces (like a pizza)
        x = math.sin(2*i*math.pi/divisions) * distance + center[0]
        y = math.cos(2*i*math.pi/divisions) * distance + center[1]    
        x2 = math.sin(2*(i+1)*math.pi/divisions) * distance + center[0]
        y2 = math.cos(2*(i+1)*math.pi/divisions) * distance + center[1]    
        xMid = math.sin(2*(i+.5)*math.pi/divisions) * 123 + center[0]
        yMid = math.cos(2*(i+.5)*math.pi/divisions) * 123 + center[1]
    
        top = tuple(np.array((x,y), int))
        bottom = tuple(np.array((x2,y2), int))
        midpoint = tuple(np.array((xMid,yMid), int))
    
        pt1 = center
        pt2 = top
        pt3 = bottom
        triangle_cnt = np.array( [pt1, pt2, pt3])
        
        mask = np.zeros((h,w), np.uint8)   
        cv2.drawContours(mask, [triangle_cnt], 0, (255,255,255), -1)
        
        cv2.line(img, center, top, 255, 4)
        
        cv2.line(img, center, bottom, 255, 4)
               
       
        cv2.circle(img, (int(xc),int(yc)), 5, (255, 255, 255), -1)
        
        CLRS = [(0, 0, 0), (255, 255, 0), (255, 0, 0), (255, 0, 255), (0, 255, 255), (128,0, 56), (0, 0, 255), (255, 255, 255)]
      
    
        CLR =  CLRS[i]
        # Iterate through the points in the contour
        CSlice = []
        for j in range(int(c.shape[0])):
            ctemp= c[j, :, :]
            x = ctemp[0][0]
            y = ctemp[0][1]
            CNew=[]
            # Check if the point is in the white section
            if mask[y,x] == 255:
                CNew = [x, y]
                sector_number[j] = i
            CSlice += CNew 
        CSlice = np.array(CSlice)  
        CSlice = CSlice.reshape(int(CSlice.shape[0]/2),2)
        
        # CPie += CSlice 
        CPie['layer_' + str(i)] = CSlice
        #cv2.drawContours(img, [CSlice], 0, CLR, -1)   
        # plt.figure()
        # plt.imshow(img)
        # plt.axis('off')

    # Calculation of Border feature
    
    # 1. Euclidean Distance
    
    center = (xc,yc)
    C0 = CPie.get('layer_0')
    
    eDistanceL0=[]
    for i in range(int(C0.shape[0])):
        x = C0[i][0]
        y = C0[i][1]
        eDistance_temp = math.dist([x, y], [int(xc),int(yc)])
        eDistanceL0.append(eDistance_temp)
    # print(eDistance)
    
    
    C0 = CPie.get('layer_1')
    
    eDistanceL1=[]
    for i in range(int(C0.shape[0])):
        x = C0[i][0]
        y = C0[i][1]
        eDistance_temp = math.dist([x, y], [int(xc),int(yc)])
        eDistanceL1.append(eDistance_temp)
        
        
    C0 = CPie.get('layer_2')
    
    eDistanceL2=[]
    for i in range(int(C0.shape[0])):
        x = C0[i][0]
        y = C0[i][1]
        eDistance_temp = math.dist([x, y], [int(xc),int(yc)])
        eDistanceL2.append(eDistance_temp)
        
        
    C0 = CPie.get('layer_3')
    
    eDistanceL3=[]
    for i in range(int(C0.shape[0])):
        x = C0[i][0]
        y = C0[i][1]
        eDistance_temp = math.dist([x, y], [int(xc),int(yc)])
        eDistanceL3.append(eDistance_temp)
        
        
        
    C0 = CPie.get('layer_4')
    
    eDistanceL4=[]
    for i in range(int(C0.shape[0])):
        x = C0[i][0]
        y = C0[i][1]
        eDistance_temp = math.dist([x, y], [int(xc),int(yc)])
        eDistanceL4.append(eDistance_temp)
    
    
    C0 = CPie.get('layer_5')
    
    eDistanceL5=[]
    for i in range(int(C0.shape[0])):
        x = C0[i][0]
        y = C0[i][1]
        eDistance_temp = math.dist([x, y], [int(xc),int(yc)])
        eDistanceL5.append(eDistance_temp)
        
        
    C0 = CPie.get('layer_6')
    
    eDistanceL6=[]
    for i in range(int(C0.shape[0])):
        x = C0[i][0]
        y = C0[i][1]
        eDistance_temp = math.dist([x, y], [int(xc),int(yc)])
        eDistanceL6.append(eDistance_temp)
        
        
        
    C0 = CPie.get('layer_7')
    
    eDistanceL7=[]
    for i in range(int(C0.shape[0])):
        x = C0[i][0]
        y = C0[i][1]
        eDistance_temp = math.dist([x, y], [int(xc),int(yc)])
        eDistanceL7.append(eDistance_temp)
        

    # stdev([S0, S1, S2, S3, S4, S5, S6, S7])
    
    Std0 = stdev(eDistanceL0) 
    # print(Std0) 
    Std1 = stdev(eDistanceL1) 
    # print(Std1)  
    Std2 = stdev(eDistanceL2) 
    # print(Std2) 
    Std3 = stdev(eDistanceL3) 
    # print(Std3) 
    Std4 = stdev(eDistanceL4) 
    # print(Std4) 
    Std5 = stdev(eDistanceL5) 
    # print(Std5) 
    Std6 = stdev(eDistanceL6) 
    # print(Std6) 
    Std7 = stdev(eDistanceL7) 
    # print(Std7) 
           
    STD = [Std0, Std1, Std2, Std3, Std4, Std5, Std6, Std7] 
    # 3. Threshold and Calculate B value.

    BScore =0
    
    for std in range(len(STD)):
   
        ### This is the threshold to be ste for border score
        
        if STD[std]> 13:
            BScore = BScore+1
        
    # print(BScore)
    
    # fig = plt.figure() # Creates a new figure
    # plt.figure()
    # plt.imshow(img)
    # plt.axis('off')
    # plt.rcParams["font.size"] = "20"
    # plt.title('Border:  %i' %BScore)
    
    
    # Border Saliency score:
    

    E_OUT = []
    E_IN = []
         
    for pt in range(CT.shape[0]):
        
        
        # point = tuple(CT[pt, :])
            
        targetY = CT[pt, 1] # point[1]
        gunY = int(yc)
        targetX = CT[pt, 0] #point[0]
        gunX = int(xc)
        
        myradians = math.atan2(targetY-gunY, targetX-gunX)
        mydegrees = math.degrees(myradians)
        
        angle = mydegrees
        
        if sector_number[pt] == 0:
           length = Std0 
        elif sector_number[pt] == 1:
            length = Std1
        elif sector_number[pt] == 2:
            length = Std2
        elif sector_number[pt] == 3:
            length = Std3
        elif sector_number[pt] == 4:
            length = Std4
        elif sector_number[pt] == 5:
            length = Std5
        elif sector_number[pt] == 6:
            length = Std6
        elif sector_number[pt] == 7:
            length = Std7
        
        Ey = targetY + length * math.sin(math.radians(angle))
        Ex = targetX + length * math.cos(math.radians(angle))

        E_OUT_TEMP = [Ex, Ey]
        E_OUT += E_OUT_TEMP

        # plt.plot(Ex, Ey, marker='v', color="black")
            
        length = -length
        Ey = targetY + length * math.sin(math.radians(angle))
        Ex = targetX + length * math.cos(math.radians(angle))
        
        E_IN_TEMP = [Ex, Ey]
        E_IN += E_IN_TEMP

        # plt.plot(Ex, Ey, marker='v', color="red")
        # plt.imshow(img_border)
        # plt.axis('off')
        
        
    E_OUT = np.array(E_OUT)
    E_IN = np.array(E_IN)
    E_OUT = E_OUT.reshape(int(E_OUT.shape[0]/2),2)
    E_IN = E_IN.reshape(int(E_IN.shape[0]/2),2) 
    
    ##########################################
    
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)  
    E_OUT = np.int0(E_OUT) 
    E_IN = np.int0(E_IN)
    
    img_border1 = cv2.drawContours(mask, [E_OUT], 0, (255,255,255), -1)
    
    # plt.imshow(img_border1)
    # plt.axis('off')
    
    img_border2 = cv2.drawContours(mask, [E_IN], 0, (128,128,0), -1)
    # plt.imshow(img_border2)
    # plt.axis('off')
    
    # RING = np.where(img_border2 == 255)
    boolArr = (img_border2 == 255)
    
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8) 
    
    ring= []
    for m in range(int(boolArr.shape[0])):
        for n in range(int(boolArr.shape[1])):
            C = boolArr[m, n]
            CNew = []
            if C==True:
               CNew=[m, n]
               mask[m, n]= 1
            ring += CNew 
        
    ring = np.array(ring)  
    ring = ring.reshape(int(ring.shape[0]/2),2)
    
    # mask = np.zeros((224, 224), np.uint8) 
    # plt.figure()
    # plt.imshow(mask)
    # plt.axis('off')
    ratio = 0.8
    width = mask.shape[0]
    height = mask.shape[1]
    new_size = ratio * min(width, height)
    img = trF.center_crop(torch.tensor(mask), new_size)
    PIL_image = Image.fromarray(np.uint8(img.numpy()))
    
    # PIL_image = Image.fromarray(np.uint8(mask))

    TS = transformations(PIL_image)
    
    IM = TS.numpy()
    IM = np.moveaxis(IM, 0, -1)
    cropped_BM=IM[:, :, :]    
    
    return BScore, cropped_BM

####
def ColorScore(mn, data_images, data_images_lesion):
    
    
    #########################################################
    IM_Original = data_images[mn]
    IM_Original = cv2.cvtColor(IM_Original,cv2.COLOR_BGR2RGB)
    IM_Original = np.moveaxis(IM_Original, 2, 0)
    IM_Original = torch.tensor(IM_Original)
    IM_Original = IM_Original.unsqueeze(0)
    # I_extract = utils.make_grid(IM_Original, nrow=1, normalize=False, scale_each=True)
    I_extract = utils.make_grid(IM_Original.double(), nrow=1, normalize=True, scale_each=True)
    
    # I = II[mn, :, :, :].unsqueeze(0)
    # I_extract = utils.make_grid(I, nrow=1, normalize=True, scale_each=True)
    # IM = I_extract
    IM_Original = I_extract.numpy()
    IM_Original = np.moveaxis(IM_Original, 0, -1)
    
    # plt.figure()
    # plt.imshow(IM_Original)
    # plt.axis('off')
    
    img = data_images_lesion[mn]
    plt.imsave('OriginalImage.jpg',img)
        
    img = cv2.imread('OriginalImage.jpg')
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    fg = gray.copy()
    from skimage.color import rgb2lab, lab2rgb
    Xsub_lab = rgb2lab(IM_Original)
    # Xsub_lab = rgb2lab(img)
    # plotMinMax(Xsub_lab,labels=["L","A","B"]) 
    
    #########################################################
    # IM_Original = data_images[mn]
    # IM_Original = cv2.cvtColor(IM_Original,cv2.COLOR_BGR2RGB)
    # img = data_images_lesion[mn]
    # # img=IM_Original[:, :, :]
    # # plt.figure()
    # # plt.imshow(IM_Original)
    # # plt.axis('off')
    # plt.imsave('OriginalImage.jpg',img)
        
    # img = cv2.imread('OriginalImage.jpg')
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # fg = gray.copy()
    # from skimage.color import rgb2lab, lab2rgb
    # Xsub_lab = rgb2lab(IM_Original)
    ###########################################################
    
    Bool_fg = fg == 255

    # Red Color
    point_a = np.array((54.29,80.81,69.89))
   
    E_Temp = np.sqrt(np.sum((Xsub_lab[:, :, :] - point_a) ** 2, axis=2))
    E_Red = np.ones((img.shape[0], img.shape[1]))*1000000
        
    E_Red[Bool_fg]= E_Temp[Bool_fg] 
    
    # Black Color
    
    # 	[0.06, 0.27, 0.10] − [39.91, 30.23, 22.10]
    
    point_a = np.array((0.06, 0.27, 0.10))
    
    E_Temp = np.sqrt(np.sum((Xsub_lab[:, :, :] - point_a) ** 2, axis=2))
    E_Black = np.ones((img.shape[0], img.shape[1]))*1000000
        
    E_Black[Bool_fg]= E_Temp[Bool_fg] 
    
        
    # White Color
    
    point_a = np.array((100,0,0))
    
    E_Temp = np.sqrt(np.sum((Xsub_lab[:, :, :] - point_a) ** 2, axis=2))
    E_White = np.ones((img.shape[0], img.shape[1]))*1000000
        
    E_White[Bool_fg]= E_Temp[Bool_fg] 
  
                
    # Blue-gray Color
    
    point_a = np.array((50.28, -30.14, -11.96))
    
    E_Temp = np.sqrt(np.sum((Xsub_lab[:, :, :] - point_a) ** 2, axis=2))
    E_BlueGray = np.ones((img.shape[0], img.shape[1]))*1000000
        
    E_BlueGray[Bool_fg]= E_Temp[Bool_fg]
    
    
    # Dark Brown Color
    # [14.32, 6.85, 6.96] − [47.57, 27.14, 46.81]
    
    point_a = np.array((14.32, 6.85, 6.96))
    
    E_Temp = np.sqrt(np.sum((Xsub_lab[:, :, :] - point_a) ** 2, axis=2))
    E_DarkBrown = np.ones((img.shape[0], img.shape[1]))*1000000
        
    E_DarkBrown[Bool_fg]= E_Temp[Bool_fg]
    
                
    # Light Brown Color
    # [47.94, 11.89, 19.86] − [71.65, 44.81, 64.78]
    
    point_a = np.array((47.94, 11.89, 19.86)) # change here again
    
    E_Temp = np.sqrt(np.sum((Xsub_lab[:, :, :] - point_a) ** 2, axis=2))
    E_LightBrown = np.ones((img.shape[0], img.shape[1]))*1000000
        
    E_LightBrown[Bool_fg]= E_Temp[Bool_fg]
    
    
     # Blue-gray Color
    
    point_a = np.array((50.28, -30.14, -11.96))
    
    E_Temp = np.sqrt(np.sum((Xsub_lab[:, :, :] - point_a) ** 2, axis=2))
    E_BlueGray = np.ones((img.shape[0], img.shape[1]))*1000000
        
    E_BlueGray[Bool_fg]= E_Temp[Bool_fg]
    
    ################# Thresholding
    
        
    Color_Score = 0
    TH=10    
    # Red Color
    
    #E_Red = E_Red_Images[img_number].copy()   

    BoolE = E_Red< TH
    
    if np.any(BoolE) == True:
        Color_Score = Color_Score + 1
    
    E_Red = np.zeros((img.shape[0], img.shape[1]))    
    E_Red[BoolE] = 1


    # White Color    
    #E_White = E_White_Images[img_number].copy()   

    BoolE = E_White< TH
    
    if np.any(BoolE) == True:
        Color_Score = Color_Score + 1
        
    E_White = np.zeros((img.shape[0], img.shape[1]))    
    E_White[BoolE] = 1    
    
    # Black Color    
    #E_Black = E_Black_Images[img_number].copy()   

    BoolE = E_Black< TH
    
    if np.any(BoolE) == True:
        Color_Score = Color_Score + 1
    
    E_Black = np.zeros((img.shape[0], img.shape[1]))    
    E_Black[BoolE] = 1
    
    # LightBrown Color    
    #E_LightBrown = E_LightBrown_Images[img_number].copy()   

    BoolE = E_LightBrown< TH
    
    if np.any(BoolE) == True:
        Color_Score = Color_Score + 1
    
    E_LightBrown = np.zeros((img.shape[0], img.shape[1]))    
    E_LightBrown[BoolE] = 1    
        
    # DarkBrown Color    
    #E_DarkBrown = E_DarkBrown_Images[img_number].copy()   

    BoolE = E_DarkBrown< TH
    
    if np.any(BoolE) == True:
        Color_Score = Color_Score + 1
    
    E_DarkBrown = np.zeros((img.shape[0], img.shape[1]))    
    E_DarkBrown[BoolE] = 1      
        
    # BlueGray Color    
    #E_BlueGray = E_BlueGray_Images[img_number].copy()   

    BoolE = E_BlueGray< TH
    
    if np.any(BoolE) == True:
        Color_Score = Color_Score + 1
        
    E_BlueGray = np.zeros((img.shape[0], img.shape[1]))    
    E_BlueGray[BoolE] = 1   
    
    #print(Color_Score)
    
    Color_Mask = np.zeros((img.shape[0], img.shape[1]))
    Color_Mask = E_Red+ E_White+ E_Black + E_LightBrown + E_DarkBrown + E_BlueGray
    Color_Mask[Color_Mask>0] = 1 
    
    # for i in range(int(Color_Mask.shape[0])): 
    #     for j in range(int(Color_Mask.shape[1])): 
    #         if Color_Mask[i, j]> 0:
    #             Color_Mask[i, j]=1
    #         else:
    #             Color_Mask[i, j]=0 
               
    # Color_Mask_New = E_Red+ E_White+ E_Black + E_LightBrown + E_DarkBrown + E_BlueGray 
    # Color_Mask_New[Color_Mask_New>0] = 1         
    # # (Color_Mask==Color_Mask_New).all()
    
    ratio = 0.8
    width = Color_Mask.shape[0]
    height = Color_Mask.shape[1]
    new_size = ratio * min(width, height)
    img = trF.center_crop(torch.tensor(Color_Mask), new_size)
    PIL_image = Image.fromarray(np.uint8(img.numpy()))
    
    #PIL_image = Image.fromarray(np.uint8(Color_Mask))

    TS = transformations(PIL_image)
    
    IM = TS.numpy()
    IM = np.moveaxis(IM, 0, -1)
    cropped_CM=IM[:, :, :] 
               
    return Color_Score, cropped_CM


###############################################

import torch

data_old = torch.load('features_train_final.pt', map_location ='cpu')

OriginalImagesCropped = data_old['original_images']

preprocessed_images = data_old['preprocessed_images']

segmented_images = []
for mn in range(len(data_images_lesion)):# len(data_images_lesion)
    
    img_lm = data_images_lesion[mn]

    ratio = 0.8
    width = img_lm.shape[0]
    height = img_lm.shape[1]
    new_size = ratio * min(width, height)
    img = trF.center_crop(torch.tensor(img_lm), new_size)
    PIL_image = Image.fromarray(np.uint8(img.numpy()))

    TS = transformations(PIL_image)
    
    IM = TS.numpy()
    IM = np.moveaxis(IM, 0, -1)
    cropped_LM = IM[:, :, :] 
    
    segmented_images.append(cropped_LM)
               

########################################################
# Saving scores in data structures

Asymmetry_Scores = []
#Asymmetry_FeatureMask = []
Cropped_AMask = []
Border_Scores = []
#Border_FeatureMask = []
Cropped_BMask = []
Color_Scores = []
#Color_FeatureMask = []
Cropped_CMask = []
ImageShape = []
ELP =[]
     
for mn in range(len(data_images)): # len(data_images)
    
    print("Computation of ABC scores for image number %d " % mn) 
    
    try:
        AScore, AM, E, IS = AsymmetryScore(mn, data_images_lesion)
        
        # Asymmetry_Scores.append(AScore)
        
        # Asymmetry_FeatureMask.append(AMask)
    except:
        AScore = []
        #AMask = []
        AM = []
        E= []
        IS=[]
        
    try:
        BScore, BM = BorderScore(mn, data_images_lesion)
        
        # Border_Scores.append(BScore)
        
        # Border_FeatureMask.append(BMask)
    except:
        BScore = []
        #BMask = []
        BM = []
        
    try:
        CScore, CM = ColorScore(mn, data_images, data_images_lesion)
        
        # Color_Scores.append(CScore)
        
        # Color_FeatureMask.append(CMask)
    except:
        CScore = []
        #CMask = []  
        CM = []
       

    Asymmetry_Scores.append(AScore)
    
    #Asymmetry_FeatureMask.append(AMask)
    
    Cropped_AMask.append(AM)
    
    ELP.append(E)
    
    ImageShape.append(IS)
    
    Border_Scores.append(BScore)
    
    #Border_FeatureMask.append(BMask)
    
    Cropped_BMask.append(BM)
    
    Color_Scores.append(CScore)
    
    #Color_FeatureMask.append(CMask)
    
    Cropped_CMask.append(CM)

################################################

torch.save({
    'original_images': OriginalImagesCropped,
    'segmented_images': segmented_images,
    'preprocessed_images': preprocessed_images,
    'DNN_predictions': predictions,
    'DNN_probabilities': prob_st,
    'ground_truth': Labels,
    'Asymmetry_Scores': Asymmetry_Scores,
    'Cropped_AMask': Cropped_AMask,
    'Border_Scores': Border_Scores,
    'Cropped_BMask': Cropped_BMask,
    'Color_Scores': Color_Scores,
    'Cropped_CMask': Cropped_CMask,
    'Ellipse': ELP,
    'ImageShape': ImageShape
}, 'features_train_v2.pt')


# ###########################################################################
import torch
data_F = torch.load('features_test_v2.pt', map_location ='cpu')

##########################################################
# plt.figure()
# plt.imshow(data_images[mn])
# plt.axis('off')
# plt.figure()
# plt.imshow(data_images_lesion[mn])
# plt.axis('off')
# plt.figure()
# plt.imshow(segmented_images[mn])
# plt.axis('off')
# plt.figure()
# plt.imshow(data_F['Cropped_AMask'][mn])
# plt.axis('off')
# plt.figure()
# plt.imshow(data_F['Cropped_BMask'][mn])
# plt.axis('off')
# plt.figure()
# plt.imshow(data_F['Cropped_CMask'][mn])
# plt.axis('off')

# ##################################################################
plt.figure()
plt.imshow(data_F['original_images'][mn])
plt.axis('off')

