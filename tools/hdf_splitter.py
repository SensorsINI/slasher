"""HDF Splitter.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""
from __future__ import print_function, absolute_import
import argparse
import os

import h5py

import spiker
from spiker import log

logger = log.get_logger("hdf5-splitter", log.INFO)


def prepare_ds(data_name):
    """Prepare training dataset for driving by taking raw HDF5 recordings."""

    hdf5_path = os.path.join(
        spiker.SPIKER_DATA, "rosbag", data_name)
    #  write to new file
    hdf5_path_train = hdf5_path[:-5]+"-train.hdf5"
    hdf5_path_test = hdf5_path[:-5]+"-test.hdf5"

    dataset = h5py.File(hdf5_path, "r")
    dataset_train = h5py.File(hdf5_path_train, "w")
    dataset_test = h5py.File(hdf5_path_test, "w")

    images = dataset["dvs_bind"][()]
    pwm = dataset["pwm"][()]

    num_images = images.shape[0]
    num_train = int(num_images*0.7)

    dataset_train.create_dataset(
        name="dvs_bind",
        data=images[:num_train:],
        dtype="float32")
    dataset_train.create_dataset(
        name="pwm",
        data=pwm[:num_train:],
        dtype="float32")

    dataset_train.close()

    dataset_test.create_dataset(
        name="dvs_bind",
        data=images[num_train:],
        dtype="float32")
    dataset_test.create_dataset(
        name="pwm",
        data=pwm[num_train:],
        dtype="float32")

    dataset_test.close()

    dataset.close()


if __name__ == '__main__':
    # An argument parser for the program
    parser = argparse.ArgumentParser(description="Prepare driving dataset")
    parser.add_argument("--data-name", "-n", type=str,
                        default="",
                        help="name of your dataset in spikeres folder")
    args = parser.parse_args()
    prepare_ds(**vars(args))
