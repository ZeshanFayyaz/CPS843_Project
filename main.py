import os

# !/usr/bin/env python
# coding: utf-8

# In[1]:

import cv2
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Bidirectional, TimeDistributed, Conv2D
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from keras.callbacks import ModelCheckpoint

# In[2]:


noise_val = 70
image_size = 256

datadir_noisy = "/home/zeshanfayyaz/LIA/RAIN_TRAIN/datasets/Rain13k/train/input/"
data_clean = "/home/zeshanfayyaz/LIA/RAIN_TRAIN/datasets/Rain13k/train/target/"

test_output = "/home/zeshanfayyaz/LIA/CPS/"


# In[3]:


def create_data(datadir):
    training_validation_data = []
    for img in os.listdir(datadir):
        try:
            img_array = cv2.imread(os.path.join(datadir, img))
            new_array = cv2.resize(img_array, (image_size, image_size))
            new_array.astype('float32')
            training_validation_data.append([new_array, 0])
        except Exception as e:
            pass
    return training_validation_data


# In[4]:


# We have previously read all images from datadir as training + validation images, here we perform the split
# Default value for split is 85% training and 15% validation
def training_validation_split(training_validation, split=0.85):
    print("Using " + str(split * 100) + "% " + "Train and " + str(100 - (split * 100)) + "% " "Validation")
    print("Total Training + Validation Length: " + str(len(training_validation)))
    numtosplit = int(split * (len(training_validation)))
    training_data = training_validation[:numtosplit]
    validation_data = training_validation[numtosplit:]
    print("Training Data Length: " + str(len(training_data)))
    print("Validation Data Length: " + str(len(validation_data)))
    return training_data, validation_data


# In[5]:


# Call this function when we want to inspect 1 image as: degraded, ground truth, and predicted
# "type" argument refers to setting the title of the subplot as "Testing Image" or "Training Image"
def create_subplots(degraded, ground_truth, predicted, type):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(11, 11))
    # Plot 1: Degraded Image
    ax1.imshow(degraded, cmap="gray")
    ax1.title.set_text('Degraded Image')

    # Plot 2: Ground Truth Image
    ax2.imshow(ground_truth, cmap="gray")
    ax2.title.set_text('Ground truth Image')

    # Plot 3: Predicted Clean Image
    ax3.imshow(predicted, cmap="gray")
    ax3.title.set_text('Predicted Image')
    if type == "test":
        fig.suptitle("Testing Image")
    elif type == "train":
        fig.suptitle("Training Image")
    return fig


def average(lst):
    return sum(lst) / len(lst)


# In[6]:


# Calculate the PSNR and SSIM metrics using sklearn built-in calculations
# All calculations are performed with respect to the ground_truth image
# For both SSIM and PSNR, we aim for a large positive difference
# If the difference is positive, our network is learning. Else, the predicted image is worse quality than degraded
def psnr_ssim_metrics(ground_truth, predicted, degraded):
    # PSNR
    psnr_degraded = peak_signal_noise_ratio(ground_truth, degraded)
    psnr_predicted = peak_signal_noise_ratio(ground_truth, predicted)
    psnr_difference = psnr_predicted - psnr_degraded
    # SSIM
    ssim_degraded = structural_similarity(ground_truth, degraded)
    ssim_predicted = structural_similarity(ground_truth, predicted)
    ssim_difference = ssim_predicted - ssim_degraded
    return psnr_degraded, psnr_predicted, psnr_difference, ssim_degraded, ssim_predicted, ssim_difference


# In[7]:


# MODEL 4 5 CNN
# Transpose the input shape replacing rows with columns and columns with rows
# Return sequences is TRUE as we want an output for every timestep, and not a "many-to-one" output
# Merge_mode is set to AVERAGE - in order to maintain dimensionality (256,256) [default is CONCAT]
def train_model(image_size):
    size_GRU = image_size * 3

    inputs = Input(shape=(image_size, image_size, 3))
    # conv1_input = tf.expand_dims(gru_output, -1)
    # cnn 5x5 kernel. relu activation
    conv1 = Conv2D(32, kernel_size=5, activation='relu', padding='same', strides=1)(inputs)
    conv2 = Conv2D(32, kernel_size=3, activation='relu', padding='same', strides=1)(conv1)
    conv3 = Conv2D(32, kernel_size=3, activation='relu', padding='same', strides=1)(conv2)
    conv4 = Conv2D(3, kernel_size=3, activation='relu', padding='same', strides=1)(conv3)
    # cnn 3x3
    # cnn 3x3.
    # subtract inputs, output of cnn => final output
    # cnn_output = tf.squeeze(conv4, [3])
    real_output = conv4

    model = Model(inputs=inputs, outputs=real_output)
    model.summary()
    print("Model Compiled")
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-5)
    model.compile(
        optimizer=opt,
        loss='mean_squared_error',
        metrics=['mse']
    )
    return model


# In[8]:


# We separate the datadir images into Training Data and Validation Data
print("Creating Training and Validation Data...")
TRAIN_target = create_data(data_clean)
TRAIN_input = create_data(datadir_noisy)

training_data_target, validation_data_target = training_validation_split(TRAIN_target)
training_data_input, validation_data_input = training_validation_split(TRAIN_input)

training_data_target = np.array(training_data_target)
training_data_input = np.array(training_data_input)
validation_data_target = np.array(validation_data_target)
validation_data_input = np.array(validation_data_input)

# In[9]:


training_target_FIXED = []

for i in range(len(training_data_target)):
    training_target_FIXED.append(training_data_target[i][0])

training_input_FIXED = []

for i in range(len(training_data_input)):
    training_input_FIXED.append(training_data_input[i][0])

validation_input_FIXED = []

for i in range(len(validation_data_input)):
    validation_input_FIXED.append(validation_data_input[i][0])

validation_target_FIXED = []

for i in range(len(validation_data_target)):
    validation_target_FIXED.append(validation_data_target[i][0])

# In[10]:


filepath = "/home/zeshanfayyaz/LIA/CPS/model_checkpoint.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callback_list = [checkpoint]

loss_train = []
loss_val = []

model = train_model(image_size)

# In[11]:


X = training_input_FIXED
X = np.array(X)
z = training_target_FIXED
z = np.array(z)
X_validation = validation_input_FIXED
X_validation = np.array(X_validation)
z_validation = validation_target_FIXED
z_validation = np.array(z_validation)

X = np.array(X)
z = np.array(z)
X_validation = np.array(X_validation)
z_validation = np.array(z_validation)
print("Reshaping Arrays... Done")


# In[12]:


def convert2RGB(image_list):
    for i in range(len(image_list)):
        image_list[i] = cv2.cvtColor(image_list[i], cv2.COLOR_RGB2BGR)


# In[13]:


convert2RGB(X)

# In[14]:


convert2RGB(z)

# In[15]:


convert2RGB(X_validation)

# In[16]:


convert2RGB(z_validation)

# In[17]:


print(X.shape)
print(z.shape)
print(X_validation.shape)
print(z_validation.shape)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(16, 13))
# Plot 1: Clean Training Image
ax1.imshow(X[0])
ax1.title.set_text('Degraded Truth')

# Plot 2: Degraded Training Image
ax2.imshow(z[0])
ax2.title.set_text('Ground Truth')

# Plot 3: Clean Validation Image
ax3.imshow(X_validation[0])
ax3.title.set_text('Degraded Truth_VAL')

# Plot 2: Degraded Validation Image
ax4.imshow(z_validation[0])
ax4.title.set_text('Ground Truth_VAL')

# In[ ]:


# Calculate loss
loss_metrics = model.fit(X, z,
                         batch_size=30,
                         epochs=200,
                         validation_data=(X_validation, z_validation),
                         callbacks=callback_list
                         )