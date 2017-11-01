"""Keras try to tackle testing dataset.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os

import numpy as np
from scipy.misc import imresize
import h5py

from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Flatten
from keras.regularizers import l2

# load dataset
DATA_PATH = os.path.join(os.environ["HOME"],
                         "share", "dataset")

train_path = os.path.join(DATA_PATH, "recording1_remap.h5")
test_path = os.path.join(DATA_PATH, "Testing_remap.h5")

train_ds = h5py.File(train_path, "r")
test_ds = h5py.File(test_path, "r")

train_x = train_ds["video/image"][()].astype("float32")
train_y = train_ds["command/steering"][()].astype("float32")

test_x = test_ds["video/image"][()].astype("float32")
test_y = test_ds["command/steering"][()].astype("float32")

ds_x = np.concatenate((train_x, test_x), axis=0)
ds_y = np.concatenate((train_y, test_y), axis=0)
ds_y = ds_y[..., np.newaxis]

ds_x_new = np.zeros((ds_x.shape[0], 48, 64, 1))
for frame_idx in xrange(ds_x.shape[0]):
    ds_x_new[frame_idx, :, :, 0] = imresize(
        ds_x[frame_idx, :, :, 0], (48, 64))
ds_x = ds_x_new

print (ds_x.shape)

# preprocessing
ds_x /= 255.
ds_x -= np.mean(ds_x, keepdims=True)
ds_y -= np.mean(ds_y, keepdims=True)

num_samples = 1400
X_train = ds_x[:num_samples]
Y_train = ds_y[:num_samples]
X_test = ds_x[num_samples:]
Y_test = ds_y[num_samples:]

# build model
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

img_input = Input(shape=input_shape)

x = Conv2D(16, (5, 5), padding="same",
           kernel_initializer="lecun_normal",
           kernel_regularizer=l2(0.0001),
           bias_initializer="zeros")(img_input)
x = BatchNormalization(axis=3)(x)
x = Activation("relu")(x)
x = MaxPool2D((2, 2))(x)

x = Conv2D(16, (5, 5), padding="same",
           kernel_initializer="lecun_normal",
           kernel_regularizer=l2(0.0001),
           bias_initializer="zeros")(x)
x = BatchNormalization(axis=3)(x)
x = Activation("relu")(x)
x = MaxPool2D((2, 2))(x)

x = Flatten()(x)

x = Dense(1024,
          kernel_initializer="lecun_normal",
          kernel_regularizer=l2(0.0001),
          bias_initializer="zeros")(x)
x = Activation("relu")(x)
x = Dense(512,
          kernel_initializer="lecun_normal",
          kernel_regularizer=l2(0.0001),
          bias_initializer="zeros")(x)
x = Activation("relu")(x)

x = Dense(1,
          kernel_initializer="lecun_normal",
          #  kernel_regularizer=l2(0.0001),
          bias_initializer="zeros")(x)

# compile model
model = Model(img_input, x)
model.summary()

#  sgd = optimizers.SGD(lr=0.0, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',
              optimizer="adam",
              metrics=["mse"])
print ("[MESSAGE] Model is compiled.")

# training
model.fit(
    x=X_train, y=Y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_test, Y_test))
