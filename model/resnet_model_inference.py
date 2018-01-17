"""Test script for ResNet model inference.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function, absolute_import
import os

import h5py
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt

import spiker
from spiker import log
from spiker.models import utils

logger = log.get_logger("ResNet - Steering - Inference", log.INFO)


def get_dataset(dataset, frame_cut, target_size=(32, 64), verbose=True):
    """Get dataset from HDF5 object."""
    aps_frames = dataset["aps"][()]/255.
    dvs_frames = dataset["dvs"][()]/16.
    steering = dataset["pwm"][:, 0][()]
    steering = (steering-1500)/500.
    data_shape = aps_frames.shape[1:]
    num_data = aps_frames.shape[0]
    # frame rescaling
    if target_size is not None:
        frames = np.zeros((num_data,)+target_size+(2,))
    else:
        frames = np.zeros((num_data,)+(data_shape[1], data_shape[2])+(2,))
    for idx in range(num_data):
        if target_size is not None:
            frames[idx, :, :, 0] = imresize(
                dvs_frames[idx, frame_cut[0][0]:-frame_cut[0][1],
                           frame_cut[1][0]:-frame_cut[1][1]], target_size)
            frames[idx, :, :, 1] = imresize(
                aps_frames[idx, frame_cut[0][0]:-frame_cut[0][1],
                           frame_cut[1][0]:-frame_cut[1][1]], target_size)
        else:
            frames[idx, :, :, 0] = dvs_frames[
                idx, frame_cut[0][0]:-frame_cut[0][1],
                frame_cut[1][0]:-frame_cut[1][1]]
            frames[idx, :, :, 1] = aps_frames[
                idx, frame_cut[0][0]:-frame_cut[0][1],
                frame_cut[1][0]:-frame_cut[1][1]]
        if verbose is True:
            if (idx+1) % 100 == 0:
                print ("[MESSAGE] %d images processed." % (idx+1))

    return frames, steering


# model path
model_path = os.path.join(
    spiker.SPIKER_EXPS,
    "resnet_model_small_new_aps_test_exp",
    "resnet_model_small_new_aps_test_exp-139-0.02.hdf5")

frame_cut = [[40, 20], [0, 1]]

# load data
data_path = os.path.join(spiker.SPIKER_DATA, "rosbag",
                         "ccw_foyer_record_12_12_17_test_exported.hdf5")
data_path_1 = os.path.join(spiker.SPIKER_DATA, "rosbag",
                           "cw_foyer_record_12_12_17_test_exported.hdf5")
logger.info("Dataset %s" % (data_path))
test_dataset = h5py.File(data_path, "r")
test_dataset_1 = h5py.File(data_path_1, "r")

test_frames, test_steering = get_dataset(
    test_dataset, frame_cut, verbose=True)
test_steering = test_dataset["pwm"][:, 0][()]
test_frames -= np.mean(test_frames, keepdims=True)

test_frames_1, test_steering_1 = get_dataset(
    test_dataset_1, frame_cut, verbose=True)
test_steering_1 = test_dataset_1["pwm"][:, 0][()]
test_frames_1 -= np.mean(test_frames_1, keepdims=True)

# rescale steering
test_dataset.close()
test_dataset_1.close()

test_frames = np.concatenate((test_frames, test_frames_1), axis=0)
test_steering = np.concatenate(
    (test_steering, test_steering_1), axis=0)

num_samples = test_frames.shape[0]
X_test = test_frames[:, :, :, 1][..., np.newaxis]
Y_test = test_steering

logger.info("Number of samples %d" % (num_samples))
logger.info("Number of test samples %d" % (X_test.shape[0]))

# load model
model = utils.keras_load_model(model_path)

model_json = model.to_json()
with open(model_path[:-5]+"-exported.json", "w") as outfile:
    outfile.write(model_json)
    outfile.close()
model.save_weights(model_path[:-5]+"-exported.h5")

Y_predicted_test = utils.keras_predict_batch(model, X_test, verbose=True)
Y_predicted_test = (Y_predicted_test*500)+1500

# plot the model
plt.figure()
plt.plot(Y_test, "r", label="groundtruth")
plt.plot(Y_predicted_test, "g", label="predicted")
plt.legend()
plt.show()
