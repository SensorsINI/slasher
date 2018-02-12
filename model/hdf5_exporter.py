"""HDF5 Exporter.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function, absolute_import
from builtins import range
import json
import argparse
import os

import numpy as np
from skimage.transform import resize
import h5py

import spiker
from spiker import log


def determine_bind_cut(bind_time, pwm_time):
    """Determine data start point.

    This function finds the first and last valid pwm and bind
    signal based on time.
    """
    if bind_time[0] <= pwm_time[0]:
        # pwm arrives after bind
        bind_head = np.nonzero(bind_time > pwm_time[0])[0][0]-1
        pwm_head = 0
    elif bind_time[0] > pwm_time[0]:
        # pwm arrives before bind
        bind_head = 0
        pwm_head = np.nonzero(pwm_time > bind_time[0])[0][0]

    if bind_time[-1] <= pwm_time[-1]:
        # pwm finish after
        bind_tail = bind_time.shape[0]
        pwm_tail = np.nonzero(pwm_time > bind_time[-1])[0][0]
    elif bind_time[-1] > pwm_time[-1]:
        # pwm finish before
        bind_tail = np.nonzero(bind_time > pwm_time[-1])[0][0]-1
        pwm_tail = pwm_time.shape[0]

    return bind_head, pwm_head, bind_tail+1, pwm_tail+1


logger = log.get_logger("hdf5-exporter", log.INFO)


def prepare_ds(data_name, out_dir, target_size, frame_cut):
    """Prepare training dataset for driving by taking raw HDF5 recordings."""
    hdf5_path = os.path.join(
        spiker.SPIKER_DATA, "rosbag", data_name)
    #  write to new file
    hdf5_path_new = data_name[:-5]+"_exported.hdf5"
    hdf5_path_new = os.path.join(
        out_dir, hdf5_path_new)

    if os.path.isfile(hdf5_path_new):
        print ("The target output file exists! You might generate this"
               " file before. If you are intended to regenerate with"
               " with different configuration, please remove this file"
               " to proceed.")
        return hdf5_path_new

    dataset = h5py.File(hdf5_path, "r")
    dataset_ex = h5py.File(hdf5_path_new, "w")

    # basic stats
    bind_time = dataset["extra/bind/bind_ts"][()]
    pwm_time = dataset["extra/pwm/pwm_ts"][()]

    # properly cut the data
    bind_head, pwm_head, bind_tail, pwm_tail = \
        determine_bind_cut(bind_time, pwm_time)

    bind_data = dataset["extra/bind/bind_data"][bind_head:bind_tail][()]
    bind_time = bind_time[bind_head:bind_tail]

    num_imgs = bind_data.shape[0]
    pwm_data = dataset["extra/pwm/pwm_data"][pwm_head:pwm_tail][()]
    pwm_time = pwm_time[pwm_head:pwm_tail]
    num_cmds = pwm_data.shape[0]

    logger.info("Number of images: %d" % (num_imgs))
    logger.info("Number of commands: %d" % (num_cmds))

    #  since pwm is sampled around 10Hz in a very stable rate
    #  use as a sync signal

    num_samples = num_cmds
    img_shape = (num_samples, )+tuple(target_size)+(2,)

    # define dataset
    bind_data_ds = dataset_ex.create_dataset(
        name="dvs_bind",
        shape=img_shape,
        dtype="float32")
    cmd_data_ds = dataset_ex.create_dataset(
        name="pwm",
        shape=(num_samples, pwm_data.shape[1]),
        dtype="float32")

    for cmd_idx in range(num_cmds):
        # current command time
        curr_cmd_time = pwm_time[cmd_idx]

        # find the closest frame
        frame = bind_data[
            np.nonzero(bind_time <= curr_cmd_time)[0][-1]].astype("float32")

        # rescale
        frame[..., 0] /= 16.
        frame[..., 1] /= 255.

        if target_size is not None:
            frame = resize(
                frame[frame_cut[0][0]:-frame_cut[0][1],
                      frame_cut[1][0]:-frame_cut[1][1], :],
                target_size,
                mode="reflect")
        else:
            frame = frame[:, frame_cut[0][0]:-frame_cut[0][1],
                          frame_cut[1][0]:-frame_cut[1][1], :]

        current_pwm = pwm_data[cmd_idx].copy()
        current_pwm[0] = (current_pwm[0]-1500)/500.

        # assign value
        cmd_data_ds[cmd_idx] = current_pwm
        bind_data_ds[cmd_idx] = frame

        logger.info("Processed %d/%d command" % (cmd_idx+1, num_cmds))

    dataset.close()
    dataset_ex.close()

    return hdf5_path_new
