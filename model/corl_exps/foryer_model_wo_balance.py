"""Experimental ResNet Keras Model for Steering.

Foyer dataset

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function, absolute_import
import os

from sacred import Experiment

import h5py
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger

import spiker
from spiker import log
from spiker.models import resnet

logger = log.get_logger("ResNet - Steering - Experiment", log.INFO)


def get_dataset(dataset, verbose=True):
    """Get dataset from HDF5 object."""
    frames = dataset["dvs_bind"][()]
    pwm = dataset["pwm"][()]
    steering = pwm[:, 0]
    throttle = pwm[:, 1]

    return frames, steering, throttle


exp = Experiment("ResNet - Steering - Experiment")

exp.add_config({
    "model_name": "",  # the model name
    "data_name": "",  # the data name
    "test_data_name": "",  # test data name
    "stages": 0,  # number of stages
    "blocks": 0,  # number of blocks of each stage
    "filter_list": [],  # number of filters per stage
    "nb_epoch": 0,  # number of training epochs
    "batch_size": 0,  # batch size
    "target_size": [],  # target size
    })


@exp.automain
def resnet_exp(model_name, data_name, test_data_name, stages,
               blocks, filter_list, nb_epoch, batch_size, target_size):
    """Perform ResNet experiment."""
    model_path = os.path.join(spiker.SPIKER_EXPS, model_name)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    else:
        raise ValueError("[MESSAGE] This experiment has been done before."
                         " Create a new config model if you need.")
    model_file_base = os.path.join(model_path, model_name)

    # print model info
    logger.info("Model Name: %s" % (model_name))
    logger.info("Number of epochs: %d" % (nb_epoch))
    logger.info("Batch Size: %d" % (batch_size))
    logger.info("Number of stages: %d" % (stages))
    logger.info("Number of blocks: %d" % (blocks))

    # load data
    data_path = os.path.join(spiker.SPIKER_DATA, "rosbag",
                             data_name)
    test_data_path = os.path.join(spiker.SPIKER_DATA, "rosbag",
                                  test_data_name)

    if not os.path.isfile(data_path):
        raise ValueError("This dataset does not exist at %s" % (data_path))
    logger.info("Dataset %s" % (data_path))
    dataset = h5py.File(data_path, "r")
    test_dataset = h5py.File(test_data_path, "r")

    train_frames, train_steering, train_throttle = get_dataset(dataset)
    test_frames, test_steering, test_throttle = get_dataset(test_dataset)

    train_frames -= np.mean(train_frames, keepdims=True)
    test_frames -= np.mean(test_frames, keepdims=True)

    # rescale steering
    dataset.close()
    test_dataset.close()

    num_samples = train_frames.shape[0]+test_frames.shape[0]
    X_train = train_frames
    Y_train = [train_steering, train_steering]
    X_test = test_frames
    Y_test = [test_steering, test_steering]

    logger.info("Number of samples %d" % (num_samples))
    logger.info("Number of train samples %d" % (X_train.shape[0]))
    logger.info("Number of test samples %d" % (X_test.shape[0]))

    # setup image shape
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

    # Build model
    model = resnet.resnet_builder(
        model_name=model_name, input_shape=input_shape,
        batch_size=batch_size,
        filter_list=filter_list, kernel_size=(3, 3),
        output_dim=1, stages=stages, blocks=blocks,
        bottleneck=False, network_type="corl")

    model.summary()

    model.compile(loss=['mean_squared_error', 'mean_squared_error'],
                  optimizer="adam",
                  metrics=["mse", "mse"])
    logger.info("Model is compiled.")

    model_file = model_file_base + "-best.hdf5"
    checkpoint = ModelCheckpoint(model_file,
                                 monitor='val_mean_squared_error',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')

    csv_his_log = os.path.join(model_path, "csv_history.log")
    csv_logger = CSVLogger(csv_his_log, append=True)

    callbacks_list = [checkpoint, csv_logger]

    # training
    model.fit(
        x=X_train, y=Y_train,
        batch_size=batch_size,
        epochs=nb_epoch,
        validation_data=(X_test, Y_test),
        callbacks=callbacks_list)
