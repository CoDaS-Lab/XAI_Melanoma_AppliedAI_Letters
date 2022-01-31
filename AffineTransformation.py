# Affine transformation


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
# import imutils
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

rootdir =  'C:/Users\harsh\Desktop\HBK008\CoDaS Lab\Project XAI\data_2016\Test'
images_benign = [cv2.imread(file) for file in glob.glob(os.path.join(rootdir, 'benign', '*.jpg'))]
images_malignant = [cv2.imread(file) for file in glob.glob(os.path.join(rootdir, 'malignant', '*.jpg'))]

data_images = images_benign + images_malignant


rootdir_lesion =  'C:/Users\harsh\Desktop\HBK008\CoDaS Lab\Project XAI\data_2016\Test_Lesion'
images_benign_lesion = [cv2.imread(file) for file in glob.glob(os.path.join(rootdir_lesion, 'benign', '*.png'))]
images_malignant_lesion = [cv2.imread(file) for file in glob.glob(os.path.join(rootdir_lesion, 'malignant', '*.png'))]

data_images_lesion = images_benign_lesion + images_malignant_lesion

data = torch.load('extract_test_final_new.pt', map_location ='cpu')

#######################
# images_extract = data['Images']

# II = []
# for i in range(len(images_extract)):
#     FF_temp = np.array(images_extract[i].cpu())
#     II.append(FF_temp)

# II = np.vstack(II)

# II = torch.Tensor(II)

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
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) 

##########################################

# Ellipse transformation
mn =0

img = data_images_lesion[mn]
# I = II[mn, :, :, :].unsqueeze(0)
# I_extract = utils.make_grid(I, nrow=1, normalize=True, scale_each=True)
# IM = I_extract
# IM_Original = I_extract.numpy()
# IM_Original = np.moveaxis(IM_Original, 0, -1)
# img=IM_Original[:, :, :]
# RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.figure()
# plt.imshow(img)
# plt.axis('off')
plt.imsave('OriginalImage.jpg',img)

img = cv2.imread('OriginalImage.jpg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

plt.figure()
plt.imshow(result)
plt.axis('off')


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
plt.figure(figsize=(8,8))
plt.imshow(img_elfit)
plt.axis('off')
plt.title("Original Image with ellipse fit")
plt.show()

# #############################################
# #Rotation

# center = (xc, yc)

# img_asym = gray.copy()
# # print(img_asym.shape)

# height = img_asym.shape[0]
# width = img_asym.shape[1]

# # using cv2.getRotationMatrix2D() to get the rotation matrix

# rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
# # rotate the image using cv2.warpAffine
# rotated_image = cv2.warpAffine(src=img_asym, M=rotate_matrix, dsize=(width, height))

# cv2.imwrite('rotated_image.jpg', rotated_image)

# img_rt = cv2.imread('rotated_image.jpg')

# plt.figure(figsize=(8,8))
# plt.imshow(img_rt)
# plt.axis('off')
# plt.title("Rotated Image")
# plt.show()


##############################

# example code

import numpy as np
from scipy import ndimage as ndi
from skimage.io import imread
from skimage.color import rgb2gray


img = plt.imread('melanoma_ellipse.jpg')
img = rgb2gray(img)
w, h = img.shape
plt.figure(figsize=(5, 5))
plt.imshow(img)
plt.axis('off')


# identity matrix
mat_identity = np.array([[1,0,0],[0,1,0],[0,0,1]])
img1 = ndi.affine_transform(img, mat_identity)
plt.figure(figsize=(5, 5))
plt.imshow(img1)
plt.axis('off')

# Reflection
mat_reflect = np.array([[1,0,0],[0,-1,0],[0,0,1]]) @ np.array([[1,0,0],[0,1,-h],[0,0,1]])
img1 = ndi.affine_transform(img, mat_reflect) # offset=(0,h)
plt.figure(figsize=(5, 5))
plt.imshow(img1)
plt.axis('off')

# Scaling
s_x, s_y = 0.75, 1.25
mat_scale = np.array([[s_x,0,0],[0,s_y,0],[0,0,1]])
img1 = ndi.affine_transform(img, mat_scale)
plt.figure(figsize=(5, 5))
plt.imshow(img1)
plt.axis('off')

# rotate
theta = np.pi/6
mat_rotate = np.array([[1,0,w/2],[0,1,h/2],[0,0,1]]) @ np.array([[np.cos(theta),np.sin(theta),0],[np.sin(theta),-np.cos(theta),0],[0,0,1]]) @ np.array([[1,0,-w/2],[0,1,-h/2],[0,0,1]])
img1 = ndi.affine_transform(img, mat_rotate)
plt.figure(figsize=(5, 5))
plt.imshow(img1)
plt.axis('off')

# shearing
lambda1 = 0.5
mat_shear = np.array([[1,lambda1,0],[lambda1,1,0],[0,0,1]])
img1 = ndi.affine_transform(img, mat_shear)
plt.figure(figsize=(5, 5))
plt.imshow(img1)
plt.axis('off')


# All 
mat_all = mat_identity @ mat_reflect @ mat_scale @ mat_rotate @ mat_shear
img1 = ndi.affine_transform(img, mat_all)
plt.figure(figsize=(5, 5))
plt.imshow(img1)
plt.axis('off')

##########

# 1. Ratio cropping the original image.
# 2. Resize to 256.
# 3. Centre cropping to 224. M?

# Do reverse order operations from cropped image and arrive at original image.
# Shift first (shift the centre of ellipse to [0, 0]), rotate major axis to x axis, 
# and scale it so both major and minor axis are 1.

# 5 parametrs of ellipse, size of image. Define funcyion with 7 parametsr and X, y of pixels.
# output x' y' which is the new coordinate.




################################################################################

# Example code

import matplotlib.pyplot as plt
import numpy as np
import string

# points a, b and, c
a, b, c, d = (0, 1, 0), (1, 0, 1), (0, -1, 2), (-1, 0, 3)

# matrix with row vectors of points
A = np.array([a, b, c, d])

# 3x3 Identity transformation matrix
I = np.eye(3)




color_lut = 'rgbc'
fig = plt.figure()
ax = plt.gca()
xs = []
ys = []
for row in A:
    output_row = I @ row
    x, y, i = output_row
    xs.append(x)
    ys.append(y)
    i = int(i) # convert float to int for indexing
    c = color_lut[i]
    plt.scatter(x, y, color=c)
    plt.text(x + 0.15, y, f"{string.ascii_letters[i]}")
xs.append(xs[0])
ys.append(ys[0])
plt.plot(xs, ys, color="gray", linestyle='dotted')
ax.set_xticks(np.arange(-2.5, 3, 0.5))
ax.set_yticks(np.arange(-2.5, 3, 0.5))
plt.grid()
plt.show()


# create the scaling transformation matrix
T_s = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])

fig = plt.figure()
ax = plt.gca()
xs_s = []
ys_s = []
for row in A:
    output_row = T_s @ row
    x, y, i = row
    x_s, y_s, i_s = output_row
    xs_s.append(x_s)
    ys_s.append(y_s)
    i, i_s = int(i), int(i_s) # convert float to int for indexing
    c, c_s = color_lut[i], color_lut[i_s] # these are the same but, its good to be explicit
    plt.scatter(x, y, color=c)
    plt.scatter(x_s, y_s, color=c_s)
    plt.text(x + 0.15, y, f"{string.ascii_letters[int(i)]}")
    plt.text(x_s + 0.15, y_s, f"{string.ascii_letters[int(i_s)]}'")

xs_s.append(xs_s[0])
ys_s.append(ys_s[0])
plt.plot(xs, ys, color="gray", linestyle='dotted')
plt.plot(xs_s, ys_s, color="gray", linestyle='dotted')
ax.set_xticks(np.arange(-2.5, 3, 0.5))
ax.set_yticks(np.arange(-2.5, 3, 0.5))
plt.grid()
plt.show()


# create the rotation transformation matrix
T_r = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

fig = plt.figure()
ax = plt.gca()
for row in A:
    output_row = T_r @ row
    x_r, y_r, i_r = output_row
    i_r = int(i_r) # convert float to int for indexing
    c_r = color_lut[i_r] # these are the same but, its good to be explicit
    letter_r = string.ascii_letters[i_r]
    plt.scatter(x_r, y_r, color=c_r)
    plt.text(x_r + 0.15, y_r, f"{letter_r}'")

plt.plot(xs, ys, color="gray", linestyle='dotted')
ax.set_xticks(np.arange(-2.5, 3, 0.5))
ax.set_yticks(np.arange(-2.5, 3, 0.5))
plt.grid()
plt.show()


# create combined tranformation matrix
T = T_s @ T_r

fig = plt.figure()
ax = plt.gca()

xs_comb = []
ys_comb = []
for row in A:
    output_row = T @ row
    x, y, i = row
    x_comb, y_comb, i_comb = output_row
    xs_comb.append(x_comb)
    ys_comb.append(y_comb)
    i, i_comb = int(i), int(i_comb) # convert float to int for indexing
    c, c_comb = color_lut[i], color_lut[i_comb] # these are the same but, its good to be explicit
    letter, letter_comb = string.ascii_letters[i], string.ascii_letters[i_comb]
    plt.scatter(x, y, color=c)
    plt.scatter(x_comb, y_comb, color=c_comb)
    plt.text(x + 0.15 , y, f"{letter}")
    plt.text(x_comb + 0.15, y_comb, f"{letter_comb}'")
xs_comb.append(xs_comb[0])
ys_comb.append(ys_comb[0])
plt.plot(xs, ys, color="gray", linestyle='dotted')
plt.plot(xs_comb, ys_comb, color="gray", linestyle='dotted')
ax.set_xticks(np.arange(-2.5, 3, 0.5))
ax.set_yticks(np.arange(-2.5, 3, 0.5))
plt.grid()
plt.show()

#####



