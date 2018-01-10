"""Test script for ResNet model inference.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function, absolute_import
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

import spiker
from spiker import log
from spiker.models import utils

logger = log.get_logger("ResNet - Steering - Inference", log.INFO)

# model path
model_path = os.path.join(
    spiker.SPIKER_EXPS,
    "resnet_model_test_exp",
    "resnet_model_test_exp-70-0.02.hdf5")

# load data
data_path = os.path.join(spiker.SPIKER_DATA, "rosbag",
                         "ccw_foyer_record_12_12_17_test_exported.hdf5")
logger.info("Dataset %s" % (data_path))
dataset = h5py.File(data_path, "r")
aps_frames = dataset["aps"][()]/255.
dvs_frames = dataset["dvs"][()]/16.
steering = dataset["pwm"][:, 0][()]
# rescale steering
dataset.close()

frames = np.stack((dvs_frames, aps_frames), axis=-1)
logger.info(frames.shape)

frames -= np.mean(frames, keepdims=True)
num_samples = frames.shape[0]
num_train = int(num_samples*0.7)
X_train = frames[:num_train]
Y_train = steering[:num_train]
X_test = frames[num_train:]
Y_test = steering[num_train:]

del frames

logger.info("Number of samples %d" % (num_samples))
logger.info("Number of train samples %d" % (X_train.shape[0]))
logger.info("Number of test samples %d" % (X_test.shape[0]))

# load model
model = utils.keras_load_model(model_path)

Y_predicted_test = utils.keras_predict_batch(model, X_test, verbose=True)
Y_predicted_test = (Y_predicted_test*500)+1500

# plot the model
plt.figure()
plt.plot(Y_test, "r")
plt.plot(Y_predicted_test, "g")
plt.show()
