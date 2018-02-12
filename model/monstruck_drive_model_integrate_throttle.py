"""Experimental ResNet Keras Model for Steering.

- With only steering prediction
- Integrate throttle as input

NOTE: MAKE SURE YOU HAVE THE RIGHT INPUT!!!!!

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function, absolute_import
import os

from sacred import Experiment

import h5py
import numpy as np
from keras.layers import Input, Dense
from keras.layers import concatenate
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.utils import Sequence
from keras.regularizers import l2
from keras.models import Model

import spiker
from spiker import log
from spiker.models import resnet

import hdf5_exporter

logger = log.get_logger("ResNet - Steering - Experiment", log.INFO)


class MonstruckSequence(Sequence):
    def __init__(self, dataset, batch_size, mode):
        """Monstruck Sequence."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.mode = mode

    def __len__(self):
        return np.ceil(
            self.dataset["dvs_bind"].shape[0]/float(self.batch_size))

    def __getitem__(self, idx):
        # batch_x: data, batch_y: steering
        batch_x = self.dataset["dvs_bind"][
            idx*self.batch_size:(idx+1)*self.batch_size][()] \
                if self.mode == 2 else self.dataset["dvs_bind"][
            idx*self.batch_size:(idx+1)*self.batch_size][
                ()][..., self.mode][..., np.newaxis]
        batch_x_throttle = self.dataset["pwm"][
            idx*self.batch_size:(idx+1)*self.batch_size, 1][()]
        batch_y = self.dataset["pwm"][
            idx*self.batch_size:(idx+1)*self.batch_size, 0][()]

        return [np.array(batch_x), batch_x_throttle], np.array(batch_y)


exp = Experiment("ResNet - Steering - Experiment")

exp.add_config({
    "model_name": "",  # the model name
    "data_name": "",  # the data name
    "test_data_name": "",  # test data name
    "channel_id": 0,  # which channel to chose, 0: dvs, 1: aps, 2: both
    "stages": 0,  # number of stages
    "blocks": 0,  # number of blocks of each stage
    "filter_list": [],  # number of filters per stage
    "nb_epoch": 0,  # number of training epochs
    "batch_size": 0,  # batch size
    "frame_cut": [],  # frame cut from full resolution
                      # [[top, bottom], [left, right]]
    "target_size": [],  # [height, width]
    "early_stop": 0,  # early stopping parameter, 0 means no early stopping
    })


@exp.automain
def resnet_exp(model_name, data_name, test_data_name,
               channel_id, stages,
               blocks, filter_list, nb_epoch, batch_size, frame_cut,
               target_size, early_stop):
    """Perform ResNet experiment."""
    model_path = os.path.join(spiker.SPIKER_EXPS, model_name)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    else:
        raise ValueError("[MESSAGE] This experiment has been done before."
                         " Create a new config model if you need.")
    model_pic = os.path.join(model_path, model_name+"-model-pic.png")
    model_file_base = os.path.join(model_path, model_name)

    # print model info
    logger.info("Model Name: %s" % (model_name))
    logger.info("Number of epochs: %d" % (nb_epoch))
    logger.info("Batch Size: %d" % (batch_size))
    logger.info("Number of stages: %d" % (stages))
    logger.info("Number of blocks: %d" % (blocks))

    # prepare training dataset
    data_path = hdf5_exporter.prepare_ds(
        data_name, model_path, target_size, frame_cut)
    test_data_path = hdf5_exporter.prepare_ds(
        test_data_name, model_path, target_size, frame_cut)

    # open training and testing datasets
    dataset = h5py.File(data_path, "r")
    test_dataset = h5py.File(test_data_path, "r")

    num_train_samples = dataset["dvs_bind"].shape[0]
    num_test_samples = test_dataset["dvs_bind"].shape[0]

    logger.info("Number of samples %d" % (num_train_samples+num_test_samples))
    logger.info("Number of train samples %d" % (num_train_samples))
    logger.info("Number of test samples %d" % (num_test_samples))

    # setup image shape
    input_shape = (target_size[0], target_size[1], 2) if channel_id == 2 \
        else (target_size[0], target_size[1], 1)

    # Build model
    img_input, x = resnet.resnet_builder(
        model_name=model_name, input_shape=input_shape,
        batch_size=batch_size,
        filter_list=filter_list, kernel_size=(3, 3),
        output_dim=1, stages=stages, blocks=blocks,
        bottleneck=False, network_type="regress",
        conv_only=True)

    # a separate channel for throttle input
    throttle_input = Input(shape=(1,))
    throttle_output = Dense(
        filter_list[-1][-1], activation="relu")(throttle_input)
    # connect them together
    x = concatenate([x, throttle_output])

    # map to output layer
    x = Dense(1,
              kernel_initializer="he_normal",
              kernel_regularizer=l2(0.0001),
              bias_initializer="zeros",
              name="output")(x)

    model = Model([img_input, throttle_input], x, name=model_name)

    model.summary()
    plot_model(model, to_file=model_pic, show_shapes=True,
               show_layer_names=True)

    model.compile(loss='mean_squared_error',
                  optimizer="adam",
                  metrics=["mse"])
    logger.info("Model is compiled.")
    model_file = model_file_base + \
        "-{epoch:02d}-{val_mean_squared_error:.2f}.hdf5"
    checkpoint = ModelCheckpoint(model_file,
                                 monitor='val_mean_squared_error',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')

    csv_his_log = os.path.join(model_path, "csv_history.log")
    csv_logger = CSVLogger(csv_his_log, append=True)

    callbacks_list = [checkpoint, csv_logger]

    if early_stop != 0:
        callbacks_list.append(
            EarlyStopping(
                monitor="val_loss",
                patience=early_stop))

    # training
    model.fit_generator(
        MonstruckSequence(
            dataset, batch_size, channel_id),
        epochs=nb_epoch,
        callbacks=callbacks_list,
        validation_data=MonstruckSequence(
            test_dataset, batch_size, channel_id),
        shuffle=True)

    dataset.close()
    test_dataset.close()
