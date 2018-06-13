"""Summary data.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""
from __future__ import print_function, absolute_import
import os

import numpy as np
from skimage.transform import resize
import h5py

import spiker

# import train data
train_path = os.path.join(
    spiker.HOME, "data", "exps", "data",
    "INI_foyer_cw_ccw_training_60x180.hdf5")
test_path = os.path.join(
    spiker.HOME, "data", "exps", "data",
    "INI_foyer_cw_ccw_testing_60x180.hdf5")

train_export_path = os.path.join(
    spiker.HOME, "data", "exps", "data",
    "INI_foyer_cw_ccw_training_30x90.hdf5")
test_export_path = os.path.join(
    spiker.HOME, "data", "exps", "data",
    "INI_foyer_cw_ccw_testing_30x90.hdf5")

train_data = h5py.File(train_path, "r")
test_data = h5py.File(test_path, "r")
train_export_data = h5py.File(train_export_path, "w")
test_export_data = h5py.File(test_export_path, "w")

train_pwm = train_data["pwm"][()]
test_pwm = test_data["pwm"][()]

train_export_data.create_dataset(
        name="pwm",
        data=train_data["pwm"][()],
        dtype=np.float32)
test_export_data.create_dataset(
        name="pwm",
        data=test_data["pwm"][()],
        dtype=np.float32)

num_train_images = train_data["dvs_bind"].shape[0]
num_test_images = test_data["dvs_bind"].shape[0]
train_dvs_bind = train_export_data.create_dataset(
    name="dvs_bind",
    shape=(num_train_images, 30, 90, 2),
    dtype="float32")
test_dvs_bind = test_export_data.create_dataset(
    name="dvs_bind",
    shape=(num_test_images, 30, 90, 2),
    dtype="float32")

for idx in xrange(num_train_images):
    frame = train_data["dvs_bind"][idx][()]

    frame = resize(
        frame,
        (30, 90),
        mode="reflect")
    train_dvs_bind[idx] = frame

    print ("Processing %d/%d image" % (idx+1, num_train_images))

for idx in xrange(num_test_images):
    frame = test_data["dvs_bind"][idx][()]

    frame = resize(
        frame,
        (30, 90),
        mode="reflect")
    test_dvs_bind[idx] = frame

    print ("Processing %d/%d image" % (idx+1, num_test_images))


train_data.close()
test_data.close()
train_export_data.close()
test_export_data.close()
