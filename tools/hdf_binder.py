"""Bind multiple HDF5 driving files into one.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function

from builtins import range
import json
import argparse
import os

import numpy as np
import h5py

import spiker
from spiker import log


def bind_hdf5s(config_name):
    """Binding HDF5s, use config from a JSON file."""
    json_path = os.path.join(
        spiker.SPIKER_DATA, "rosbag", config_name)
    # read json path
    with open(json_path, "r") as f:
        config = json.load(f)
        f.close()

    binded_name = os.path.join(
        spiker.SPIKER_DATA, "rosbag", config["bind_name"])

    binded_data = h5py.File(binded_name, "w")

    # define datasets
    frame_ds = binded_data.create_dataset(
        name="dvs_bind",
        shape=(0,)+config["img_shape"]+(2,),
        maxshape=(None,)+config["img_shape"]+(2,),
        dtype="uint16")
    pwm_ds = binded_data.create_dataset(
        name="pwm",
        shape=(0, 3),
        maxshape=(None, 3),
        dtype="uint16")

    # iterate
    for file_name in config["file_list"]:
        # open file
        file_path = os.path.join(
            spiker.SPIKER_DATA, "rosbag", file_name)
        hdf5_file = h5py.File(file_path, "r")

        # dump data to binded data
        num_samples = hdf5_file["dvs_bind"].shape[0]
        resized_shape = frame_ds.shape[0]+num_samples

        frame_ds.resize(resized_shape, axis=0)
        pwm_ds.resize(resized_shape, axis=0)

        frame_ds[-num_samples:] = hdf5_file["dvs_bind"][()]
        pwm_ds[-num_samples:] = hdf5_file["pwm"][()]

        # close file
        hdf5_file.close()

    binded_data.close()



if __name__ == '__main__':
    # An argument parser for the program
    parser = argparse.ArgumentParser(description="Binding HDF5s")
    parser.add_argument("--config-name", "-n", type=str,
                        default="",
                        help="name of your dataset in spikeres folder")
    args = parser.parse_args()
    prepare_ds(**vars(args))
