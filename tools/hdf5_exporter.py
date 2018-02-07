"""HDF5 Exporter.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function, absolute_import
from builtins import range
import os

import cv2
import numpy as np
import h5py

import spiker
from spiker import log


def find_dvs_time_idx(dataset, time, idx_base=0, mode="pre", step=16):
    """Find DVS index that is pre or post of given time.

    assume time at idx_base is smaller than time given
    """
    curr_idx = idx_base
    while (dataset["dvs/event_ts"][curr_idx] < time):
        if curr_idx+step < dataset["dvs/event_ts"].shape[0]-1:
            curr_idx += step
        else:
            break

    if mode == "pre":
        return curr_idx-step
    else:
        return curr_idx


def determine_aps_cut(aps_time, pwm_time):
    """Determine data start point.

    This function finds the first and last valid pwm and aps
    signal based on time.
    """
    if aps_time[0] <= pwm_time[0]:
        # pwm arrives after aps
        aps_head = np.nonzero(aps_time > pwm_time[0])[0][0]-1
        pwm_head = 0
    elif aps_time[0] > pwm_time[0]:
        # pwm arrives before aps
        aps_head = 0
        pwm_head = np.nonzero(pwm_time > aps_time[0])[0][0]

    if aps_time[-1] <= pwm_time[-1]:
        # pwm finish after
        aps_tail = aps_time.shape[0]
        pwm_tail = np.nonzero(pwm_time > aps_time[-1])[0][0]-1
    elif aps_time[-1] > pwm_time[-1]:
        # pwm finish before
        aps_tail = np.nonzero(aps_time > pwm_time[-1])[0][0]
        pwm_tail = pwm_time.shape[0]

    return aps_head, pwm_head, aps_tail, pwm_tail


logger = log.get_logger("hdf5-exporter", log.INFO)

dvs_bin_size = 100  # ms
clip_value = 8

hdf5_path = os.path.join(
    spiker.SPIKER_DATA, "rosbag",
    "monstruck_rec_2018-01-19_indoor_cw_speeddrive.hdf5")
#  write to new file
hdf5_path_new = hdf5_path[:-5]+"_exported.hdf5"

dataset = h5py.File(hdf5_path, "r")

# basic stats
aps_time = dataset["aps/aps_ts"][()]
print (aps_time[0])
print (aps_time[346])
pwm_time = dataset["extra/pwm/pwm_ts"][()]
print (pwm_time)

aps_head, pwm_head, aps_tail, pwm_tail = determine_aps_cut(aps_time, pwm_time)

aps_data = dataset["aps/aps_data"][aps_head:aps_tail][()]
aps_time = aps_time[aps_head:aps_tail]
print (aps_time[0])
print (aps_time[346])
num_imgs = aps_data.shape[0]
img_shape = (aps_data.shape[1], aps_data.shape[2])
pwm_data = dataset["extra/pwm/pwm_data"][pwm_head:pwm_tail][()]
pwm_time = pwm_time[pwm_head:pwm_tail]
num_cmds = pwm_data.shape[0]

logger.info(num_imgs)
logger.info(num_cmds)

#  since pwm is sampled around 10Hz in a very stable rate
#  use as a sync signal

num_samples = max(num_cmds, num_imgs)

aps_data_new = np.zeros((num_samples, aps_data.shape[1], aps_data.shape[2]),
                        dtype=np.uint8)
dvs_data_new = np.zeros((num_samples, aps_data.shape[1], aps_data.shape[2]),
                        dtype=np.uint8)
pwm_data_new = np.zeros((num_samples, pwm_data.shape[1]), dtype=np.float32)
pwm_data_new[0] = pwm_data[0]
aps_data_new[0] = aps_data[0]
# fastforward to current time
curr_dvs_idx = find_dvs_time_idx(dataset, aps_time[0], mode="post")
for cmd_idx in range(1, num_cmds):
    # current command time
    curr_cmd_time = pwm_time[cmd_idx]
    # previous command time
    prev_cmd_time = pwm_time[cmd_idx-1]

    # find all frames between two cmd
    frame_idxs = np.nonzero(
        (prev_cmd_time < aps_time)*(aps_time < curr_cmd_time))[0]
    # assign data
    if frame_idxs.shape[0] == 0:
        # no data, frame rate too low
        pass
    else:
        # there is frame(s) between two command
        for idx in range(frame_idxs.shape[0]):
            if idx < frame_idxs.shape[0]-1:
                pwm_data_new[frame_idxs[idx]] = \
                    (pwm_data[cmd_idx]+pwm_data[cmd_idx-1])/2
                aps_data_new[frame_idxs[idx]] = aps_data[frame_idxs[idx]]
            else:
                pwm_data_new[frame_idxs[-1]] = pwm_data[cmd_idx]
                aps_data_new[frame_idxs[-1]] = aps_data[frame_idxs[-1]]
            # make dvs between current and next frame
            if frame_idxs[idx] < aps_time.shape[0]-1:
                bin_size = min(
                    (aps_time[frame_idxs[idx]] -
                     aps_time[frame_idxs[idx]-1])/1e3,
                    dvs_bin_size)
            else:
                bin_size = dvs_bin_size
            # find event range and bind the frame
            next_dvs_idx = find_dvs_time_idx(
                dataset, aps_time[frame_idxs[idx]-1]+bin_size*1e3,
                curr_dvs_idx)
            curr_dvs_loc = \
                dataset["dvs/event_loc"][curr_dvs_idx:next_dvs_idx][()]
            curr_dvs_pol = \
                dataset["dvs/event_pol"][curr_dvs_idx:next_dvs_idx][()]
            # fast-forward
            curr_dvs_idx = find_dvs_time_idx(
                dataset, aps_time[frame_idxs[idx]],
                next_dvs_idx, mode="post")
            # bind events
            _histrange = [(0, v) for v in img_shape]
            pol_on = (curr_dvs_pol[:] == 1)
            pol_off = np.logical_not(pol_on)
            img_on, _, _ = np.histogram2d(
                    curr_dvs_loc[pol_on, 1], curr_dvs_loc[pol_on, 0],
                    bins=img_shape, range=_histrange)
            img_off, _, _ = np.histogram2d(
                    curr_dvs_loc[pol_off, 1], curr_dvs_loc[pol_off, 0],
                    bins=img_shape, range=_histrange)
            if clip_value is not None:
                integrated_img = np.clip(
                    (img_on-img_off), -clip_value, clip_value)
            else:
                integrated_img = (img_on-img_off)
            dvs_data_new[frame_idxs[idx]] = integrated_img+clip_value

            #  cv2.imshow("aps", aps_data_new[frame_idxs[idx]])
            #  cv2.imshow("dvs", dvs_data_new[
            #      frame_idxs[idx]]/float(clip_value*2))
            #
            #  if cv2.waitKey(10) & 0xFF == ord('q'):
            #      break

        logger.info("Processed %d/%d command" % (cmd_idx, num_cmds))

dataset.close()

dataset = h5py.File(hdf5_path_new, "w")

dataset.create_dataset("aps", data=aps_data_new, dtype="uint8")
dataset.create_dataset("dvs", data=dvs_data_new, dtype="uint8")
dataset.create_dataset("pwm", data=pwm_data_new, dtype="float32")

dataset.close()
