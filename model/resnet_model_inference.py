"""Test script for ResNet model inference.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function, absolute_import
import os

import h5py
import matplotlib.pyplot as plt

import spiker
from spiker import log
from spiker.models import utils

logger = log.get_logger("ResNet - Steering - Inference", log.INFO)


# model path
model_path = os.path.join(
    spiker.SPIKER_EXPS,
    "monstruck_drive_moddel_exp",
    "monstruck_drive_moddel_exp-163-0.06.hdf5")

# load data
data_path = os.path.join(
    spiker.SPIKER_DATA, "rosbag",
    "test-walk_speed-monstruck_rec_2018-02-02-18-56-11-foyer-cw_exported.hdf5")
logger.info("Dataset %s" % (data_path))
test_dataset = h5py.File(data_path, "r")

test_frames = test_dataset["dvs_bind"][()]
test_steering = test_dataset["pwm"][:, 0][()]

# rescale steering
test_dataset.close()

num_samples = test_frames.shape[0]
X_test = test_frames
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
plt.plot((Y_test*500)+1500, "r", label="groundtruth")
plt.plot(Y_predicted_test, "g", label="predicted")
plt.legend()
plt.show()
