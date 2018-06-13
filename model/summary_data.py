"""Summary data.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""
from __future__ import print_function, absolute_import
import os

import numpy as np
import h5py

import matplotlib.pyplot as plt

import spiker

# import train data
#  train_path = os.path.join(
#      spiker.HOME, "data", "exps", "data", "jogging-train.hdf5")
#  test_path = os.path.join(
#      spiker.HOME, "data", "exps", "data", "jogging-test.hdf5")
train_path = os.path.join(
    spiker.HOME, "data", "exps", "data",
    "INI_foyer_cw_ccw_training_60x180.hdf5")
test_path = os.path.join(
    spiker.HOME, "data", "exps", "data",
    "INI_foyer_cw_ccw_testing_60x180.hdf5")

train_data = h5py.File(train_path, "r")
test_data = h5py.File(test_path, "r")

train_pwm = train_data["pwm"][()]
test_pwm = test_data["pwm"][()]

pwm = np.append(train_pwm, test_pwm, axis=0)

train_data.close()
test_data.close()

print (pwm.shape)
print (pwm[:200, 1])

# throttle
throttle = (pwm[:, 1]-1000)/1000
throttle_up = np.percentile(throttle, 75)
throttle_down = np.percentile(throttle, 25)
IQR = throttle_up-throttle_down
throttle_up += 1.5*IQR
throttle_down -= 1.5*IQR

print (throttle_down)
th_up_index = (throttle < throttle_up)
throttle = throttle[th_up_index]
th_down_index = (throttle > throttle_down)
throttle = throttle[th_down_index]
print (throttle.shape)

# Steering
steering = pwm[:, 0]
steering = steering[th_up_index]
steering = steering[th_down_index]

plt.figure()
#  plt.hist(pwm[:, 0], bins=100)
plt.boxplot(throttle)
plt.show()
