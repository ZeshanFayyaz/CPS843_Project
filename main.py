import os
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

image_size = 256
datadir_noisy = "/home/zeshanfayyaz/RAIN_TRAIN/datasets/Rain13k/train/input/"
data_clean = "/home/zeshanfayyaz/RAIN_TRAIN/datasets/Rain13k/train/target/"
test_output = "/home/zeshanfayyaz/CPS/"

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

def training_validation_split(training_validation, split=0.85):
    print("Using " + str(split * 100) + "% " + "Train and " + str(100 - (split * 100)) + "% " "Validation")
    print("Total Training + Validation Length: " + str(len(training_validation)))
    numtosplit = int(split * (len(training_validation)))
    training_data = training_validation[:numtosplit]
    validation_data = training_validation[numtosplit:]
    print("Training Data Length: " + str(len(training_data)))
    print("Validation Data Length: " + str(len(validation_data)))
    return training_data, validation_data

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

def train_model(image_size):
    size_GRU = image_size * 3
    inputs = Input(shape=(image_size, image_size, 3))
    conv1 = Conv2D(32, kernel_size=5, activation='relu', padding='same', strides=1)(inputs)
    conv2 = Conv2D(32, kernel_size=3, activation='relu', padding='same', strides=1)(conv1)
    conv3 = Conv2D(32, kernel_size=3, activation='relu', padding='same', strides=1)(conv2)
    conv4 = Conv2D(3, kernel_size=3, activation='relu', padding='same', strides=1)(conv3)
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

def convert2RGB(image_list):
    for i in range(len(image_list)):
        image_list[i] = cv2.cvtColor(image_list[i], cv2.COLOR_RGB2BGR)

print("Creating Training and Validation Data...")
TRAIN_target = create_data(data_clean)
TRAIN_input = create_data(datadir_noisy)

training_data_target, validation_data_target = training_validation_split(TRAIN_target)
training_data_input, validation_data_input = training_validation_split(TRAIN_input)

training_data_target = np.array(training_data_target)
training_data_input = np.array(training_data_input)
validation_data_target = np.array(validation_data_target)
validation_data_input = np.array(validation_data_input)

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

filepath = "/home/zeshanfayyaz/CPS/model_checkpoint.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callback_list = [checkpoint]

model = train_model(image_size)

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

convert2RGB(X)
convert2RGB(z)
convert2RGB(X_validation)
convert2RGB(z_validation)

loss_metrics = model.fit(X, z,
                         batch_size=128,
                         epochs=200,
                         validation_data=(X_validation, z_validation),
                         callbacks=callback_list
                         )
