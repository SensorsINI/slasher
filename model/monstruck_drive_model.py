"""Experimental ResNet Keras Model for Steering.

- With only steering prediction

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function, absolute_import
import os

from sacred import Experiment

import h5py
import numpy as np
from skimage.transform import resize
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.utils import Sequence

import spiker
from spiker import log
from spiker.models import resnet

logger = log.get_logger("ResNet - Steering - Experiment", log.INFO)


class MonstruckSequence(Sequence):
    def __init__(self, dataset, batch_size, frame_cut, target_size,
                 mode):
        """Monstruck Sequence."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.frame_cut = frame_cut
        self.target_size = target_size
        self.mode = mode

    def __len__(self):
        total_length = 0
        for leng in self.datasets_len:
            total_length += np.ceil(leng/float(self.batch_size))
        return total_length

    def __getitem__(self, idx):
        # batch_x: data, batch_y: steering
        batch_x = self.dataset["dvs_bind"][
            idx*self.batch_size:(idx+1)*self.batch_size][()]
        batch_y = self.dataset["pwm"][
            idx*self.batch_size:(idx+1)*self.batch_size, 0][()]

        # rescaling for x
        batch_x[..., 0] /= 16.
        batch_x[..., 1] /= 255.
        data_shape = batch_x.shape[:2]

        # rescale for steering
        batch_y = (batch_y-1500)/500.

        # preprocessing for batch data
        if self.target_size is not None:
            frames = np.zeros((self.batch_size,)+self.target_size+(2,)) \
                if self.mode == 2 else \
                np.zeros((self.batch_size,)+self.target_size+(1,))
        else:
            frames = np.zeros(
                (self.batch_size,)+(data_shape[1], data_shape[2])+(2,)) \
                if self.mode == 2 else \
                np.zeros(
                    (self.batch_size,)+(data_shape[1], data_shape[2])+(2,))

        # resize data
        if self.target_size is not None:
            frames = resize(
                batch_x[:, self.frame_cut[0][0]:-self.frame_cut[0][1],
                        self.frame_cut[1][0]:-self.frame_cut[1][1], :],
                self.target_size,
                mode="reflect")
        else:
            frames = batch_x[:, self.frame_cut[0][0]:-self.frame_cut[0][1],
                             self.frame_cut[1][0]:-self.frame_cut[1][1], :]

        return np.array(frames, dtype=np.floate32), \
            np.array(batch_y, dtype=np.float32)


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
    })


@exp.automain
def resnet_exp(model_name, data_name, test_data_name,
               channel_id, stages,
               blocks, filter_list, nb_epoch, batch_size, frame_cut,
               target_size):
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

    # load data
    data_path = os.path.join(spiker.SPIKER_DATA, "rosbag",
                             data_name)
    if test_data_name != "":
        test_data_path = os.path.join(spiker.SPIKER_DATA, "rosbag",
                                      test_data_name)

    if not os.path.isfile(data_path):
        raise ValueError("This dataset does not exist at %s" % (data_path))
    logger.info("Dataset %s" % (data_path))
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
    model = resnet.resnet_builder(
        model_name=model_name, input_shape=input_shape,
        batch_size=batch_size,
        filter_list=filter_list, kernel_size=(3, 3),
        output_dim=1, stages=stages, blocks=blocks,
        bottleneck=False, network_type="regress")

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

    # training
    model.fit_generator(
        MonstruckSequence(
            dataset, batch_size,
            frame_cut, target_size, channel_id),
        epochs=nb_epoch,
        callbacks=callbacks_list,
        validation_data=MonstruckSequence(
            test_dataset, batch_size,
            frame_cut, target_size, channel_id),
        shuffle=True)

    dataset.close()
    test_dataset.close()
