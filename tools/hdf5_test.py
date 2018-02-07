"""HDF5 Test.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function, absolute_import
from builtins import range
import os

import cv2
#  import numpy as np
import h5py
import matplotlib.pyplot as plt

import spiker
from spiker import log

logger = log.get_logger("hdf5-test", log.INFO)

hdf5_path = os.path.join(
    spiker.SPIKER_DATA, "rosbag",
    "test-walk_speed-monstruck_rec_2018-02-02-18-49-15-foyer-ccw.hdf5")

dataset = h5py.File(hdf5_path, "r")

pwm_data = dataset["/extra/pwm/pwm_data"][()]

plt.figure()
# steering data
plt.plot(pwm_data[:, 0])
plt.show()

for frame_id in range(dataset["extra/bind/bind_data"].shape[0]):
    cv2.imshow("aps", dataset["extra/bind/bind_data"][
        frame_id, :, :, 1][()])
    cv2.imshow("dvs", dataset["extra/bind/bind_data"][
        frame_id, :, :, 0][()]/float(8*2))

    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

dataset.close()
