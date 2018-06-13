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
train_path = os.path.join(
    spiker.SPIKER_DATA, "rosbag",
    "jogging_13_02_2018", "candidate-datasets", "jogging-center.hdf5")
#  test_path = os.path.join(
#      spiker.SPIKER_DATA, "rosbag", "jogging-test.hdf5")

train_data = h5py.File(train_path, "r")
#  test_data = h5py.File(test_path, "r")

pwm = train_data["pwm"][()]
#  train_pwm = train_data["pwm"][()]
#  test_pwm = test_data["pwm"][()]

#  pwm = np.append(train_pwm, test_pwm, axis=0)

train_data.close()
#  test_data.close()

print (pwm.shape)
print (pwm[:200, 1])

plt.figure()
plt.hist(pwm[:, 1], bins=200)
#  plt.boxplot(pwm[:, 1])
plt.title("jogging")
plt.show()
