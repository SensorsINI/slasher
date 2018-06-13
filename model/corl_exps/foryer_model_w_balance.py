"""Experimental ResNet Keras Model for Steering.

Jogging dataset, with balancing

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function, absolute_import
import os

from sacred import Experiment

import h5py
import numpy as np
import random
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger

import spiker
from spiker import log
from spiker.models import resnet

logger = log.get_logger("ResNet - Steering - Experiment", log.INFO)


def data_balance_gen(X_train, Y_train, batch_size=128):
    while True:
        images = np.zeros((batch_size, 30, 90, 2), dtype=np.float32)
        steerings = np.zeros((batch_size,), dtype=np.float32)
        throttles = np.zeros((batch_size,), dtype=np.float32)
        for i in range(batch_size):
            straight_count = 0
            for i in range(batch_size):
                # Select random index to use for data sample
                sample_index = random.randrange(X_train.shape[0])

                image = X_train[sample_index]
                angle = Y_train[0][sample_index]
                throttle = Y_train[1][sample_index]
                if abs(angle) < .1:
                    straight_count += 1
                if straight_count > (batch_size * .2):
                    while abs(Y_train[0][sample_index]) < .1:
                        sample_index = random.randrange(X_train.shape[0])
                        image = X_train[sample_index]
                        angle = Y_train[0][sample_index]
                        throttle = Y_train[1][sample_index]
                images[i] = image
                steerings[i] = angle
                throttles[i] = throttle

        yield images, [steerings, throttles]


def get_dataset(dataset, verbose=True):
    """Get dataset from HDF5 object."""
    frames = dataset["dvs_bind"][()]
    pwm = dataset["pwm"][()]
    steering = pwm[:, 0]
    throttle = pwm[:, 1]
    # change the throttle value from 1000-2000 to 0-1
    throttle = (throttle-1000)/1000

    # filtering out outliers from throttle
    throttle_up = np.percentile(throttle, 75)
    throttle_down = np.percentile(throttle, 25)
    IQR = throttle_up-throttle_down
    throttle_up += 1.5*IQR
    throttle_down -= 1.5*IQR

    # filter throttle
    th_up_index = (throttle < throttle_up)
    throttle = throttle[th_up_index]
    th_down_index = (throttle > throttle_down)
    throttle = throttle[th_down_index]

    # filter steering
    steering = steering[th_up_index]
    steering = steering[th_down_index]

    # filter frames
    frames = frames[th_up_index]
    frames = frames[th_down_index]

    # balancing

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
    model_path = os.path.join(spiker.HOME, "data", "exps", "models",
                              model_name)
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
    data_path = os.path.join(spiker.HOME, "data", "exps", "data",
                             data_name)
    test_data_path = os.path.join(spiker.HOME, "data", "exps", "data",
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
                  metrics=["mse"])
    logger.info("Model is compiled.")

    model_file = model_file_base + "-best.hdf5"
    checkpoint = ModelCheckpoint(model_file,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')

    csv_his_log = os.path.join(model_path, "csv_history.log")
    csv_logger = CSVLogger(csv_his_log, append=True)

    callbacks_list = [checkpoint, csv_logger]

    # training
    train_gen = data_balance_gen(X_train, Y_train, batch_size=batch_size)
    model.fit_generator(
        train_gen,
        steps_per_epoch=X_train.shape[0]//batch_size+1,
        epochs=nb_epoch,
        validation_data=(X_test, Y_test),
        callbacks=callbacks_list)
