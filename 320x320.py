#!/usr/bin/env python
# coding: utf-8

# In[262]:


import os
import cv2
import numpy as np
import random
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, GRU, Bidirectional, TimeDistributed
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Activation, BatchNormalization, Subtract, Multiply, Add, Concatenate
import pywt
import os
from skimage.transform import rescale, resize
import statistics
import numpy as np
from glob import glob
from skimage import io
from skimage.color import rgb2ycbcr
from pywt import dwt2,idwt2
import PIL.Image as Image


# In[263]:


image_size = 256

def create_data(datadir):
    training_validation_data = []
    for img in os.listdir(datadir):
        try:
            img_array = cv2.imread((os.path.join(datadir, img)))
            new_array = cv2.resize(img_array, (image_size, image_size))
            new_array.astype('float32')
            training_validation_data.append([new_array, 0])
        except Exception as e:
            pass
    return training_validation_data


# In[264]:


def myPSNR(tar_img, new_list):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(255/rmse)
    return ps


# In[265]:


def create_dataORG(TESTDIR):
    target_data = []
    for img in os.listdir(TESTDIR):
        try: 
            img_array = cv2.imread(os.path.join(TESTDIR,img))
            target_data.append([img_array])
        except Exception as e:
            pass 
    return target_data

def convert2RGB(image_list):
    for i in range (len(image_list)):
        image_list[i] = cv2.cvtColor(image_list[i], cv2.COLOR_RGB2BGR)


# In[266]:


MODEL4CNN3_13GRU8CNN = "C:\\Users\\zesha\\Desktop\\CPS_PROJECT\\model_checkpoint.h5"
our_model = load_model(MODEL4CNN3_13GRU8CNN)


# In[289]:


targetTEST = create_dataORG("C:\\Users\\zesha\\Downloads\\Test100\\target\\") #original sized images. change create_dataORG to create_data
inputTEST = create_data("C:\\Users\\zesha\\Downloads\\Test100\\input\\") #resized images
input_2 = create_dataORG("C:\\Users\\zesha\\Downloads\\Test100\\input\\")


# In[292]:


input2_FIXED = []
for i in range(len(input_2)):
    input2_FIXED.append(input_2[i][0])
    
convert2RGB(input2_FIXED)


# In[268]:


target_FIXED = []
for i in range(len(targetTEST)):
    target_FIXED.append(targetTEST[i][0])

input_FIXED = []
for i in range(len(inputTEST)):
    input_FIXED.append(inputTEST[i][0])

convert2RGB(input_FIXED)
convert2RGB(target_FIXED)


# In[294]:


target_FIXED = np.array(target_FIXED)
input_FIXED = np.array(input_FIXED)
input2_FIXED = np.array(input2_FIXED)


# In[272]:


OUTPUT = our_model.predict(input_FIXED)


# In[275]:


upsampled_images = []
for i in range(len(OUTPUT)):
    height = (target_FIXED[i].shape)[0]
    width = (target_FIXED[i].shape)[1]
    temp = cv2.resize(OUTPUT[i], dsize = (width,height), interpolation=cv2.INTER_LANCZOS4)
    temp = temp.astype(np.uint8)
    upsampled_images.append(temp)


# In[276]:


new_list = upsampled_images


# In[277]:


img_files_tar = target_FIXED
img_files_prd = new_list  


# In[278]:


print(img_files_tar[0].shape)
print(img_files_prd[0].shape)


# In[279]:


psnr_mpr= []
ssim_mpr = []
for tar_img,prd_img in zip(img_files_tar,img_files_prd):
    target_ssim = tar_img
    prd_ssim = prd_img
    tar_img = rgb2ycbcr(tar_img)[:, :, 0]
    prd_img = rgb2ycbcr(prd_img)[:, :, 0]
    PSNR_mpr = myPSNR(tar_img, prd_img)
    SSIM_mpr = structural_similarity(target_ssim, prd_ssim, multichannel=True)
    psnr_mpr.append(PSNR_mpr)
    ssim_mpr.append(SSIM_mpr)


# In[280]:


PSNR = sum(psnr_mpr)/len(psnr_mpr)
print(PSNR)


# In[281]:


SSIM = sum(ssim_mpr)/len(ssim_mpr)
print(SSIM)


# In[318]:


fig, ax = plt.subplots(2,3, sharey =True, figsize=(16,7))
#Plot 1: Clean Image
ax[0,0].imshow(input2_FIXED[5].astype('uint8'))
ax[0,0].axis('off')

#Plot 2: Predicted MPRNET Image
ax[0,1].imshow(target_FIXED[5].astype('uint8'))
ax[0,1].axis('off')

#Plot 3: Predicted OUR_NETWORK Image 
ax[0,2].imshow(img_files_prd[5].astype(np.uint8))
ax[0,2].axis('off')

#Plot 1: Clean Image
ax[1,0].imshow(input2_FIXED[12].astype('uint8'))
ax[1,0].axis('off')

#Plot 2: Predicted MPRNET Image
ax[1,1].imshow(target_FIXED[12].astype('uint8'))
ax[1,1].axis('off')

#Plot 3: Predicted OUR_NETWORK Image 
ax[1,2].imshow(img_files_prd[12].astype(np.uint8))
ax[1,2].axis('off')

