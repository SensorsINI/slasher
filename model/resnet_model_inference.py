"""Test script for ResNet model inference.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function, absolute_import
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

import spiker
from spiker import log
from spiker.models import utils

logger = log.get_logger("ResNet - Steering - Inference", log.INFO)


def collect_models(task, balance, num_trails):
    base_path = os.path.join(
        spiker.HOME, "data", "exps", "models_single")

    # curves collector
    models_path_collector = []
    for curves_idx in range(1, num_trails+1):
        # get path
        model_name = task+"_model_"+balance+"_"+str(curves_idx)
        model_path = os.path.join(
            base_path, model_name,
            model_name+"-best.hdf5")
        models_path_collector.append(model_path)

    return models_path_collector


model_paths = collect_models("foryer", "wo_balance", 10)

# load data
#  data_path = os.path.join(
#      spiker.HOME, "data", "exps", "data",
#      "INI_foyer_cw_ccw_testing_30x90.hdf5")
data_path = os.path.join(
    spiker.HOME, "data", "exps", "data",
    "foyer-test.hdf5")
#  data_path = os.path.join(
#      spiker.HOME, "data", "exps", "data", "jogging-test.hdf5")

logger.info("Dataset %s" % (data_path))
test_dataset = h5py.File(data_path, "r")

test_frames = test_dataset["dvs_bind"][()]
test_steering = test_dataset["pwm"][:, 0][()]

# rescale steering
test_dataset.close()

num_samples = test_frames.shape[0]
X_test = test_frames
Y_test = test_steering
Y_test *= 25

logger.info("Number of samples %d" % (num_samples))
logger.info("Number of test samples %d" % (X_test.shape[0]))

prediction_collector = []
for model_idx in xrange(10):
    # load model
    model = utils.keras_load_model(model_paths[model_idx])

    Y_predicted_test = utils.keras_predict_batch(model, X_test, verbose=True)
    Y_predicted_test *= 25

    prediction_collector.append(Y_predicted_test)
    del model

predictions = np.array(prediction_collector)
print (predictions.shape)
Y_mean = np.mean(predictions, axis=0)[:, 0]
Y_std = np.mean(predictions, axis=0)[:, 0]
num_steps = np.array(range(X_test.shape[0]))/30.

# plot the model
plt.figure()
plt.plot(num_steps, Y_test, lw=1,
         label="groundtruth",
         color="#5C88DAFF", ls="-", mew=5,
         alpha=0.75)

plt.plot(num_steps, prediction_collector[0], lw=2,
         label="predicted",
         color="#CC0C00FF", ls="-", mew=5)
#  plt.fill_between(num_steps, Y_mean+Y_std, Y_mean-Y_std,
#                   facecolor="#CC0C0099")

plt.xlabel("time (s)", fontsize=16)
plt.ylabel("steering angle (degree) ", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.grid()
plt.legend()
plt.show()
