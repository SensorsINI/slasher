"""Resave a Keras model into Tensorflow.

This script is a restructured version of
https://github.com/amir-abdi/keras_to_tensorflow

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import argparse

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, \
    signature_constants
import keras.backend as K
from keras.models import load_model


def keras2tf(in_model, out_model):
    """Convert Keras model to TensorFlow model."""
    # Load Keras Model
    sess = tf.Session()
    K.set_session(sess)
    K.set_learning_phase(0)
    keras_model = load_model(in_model)

    prediction_signature = \
        tf.saved_model.signature_def_utils.predict_signature_def(
            {"input": keras_model.input},
            {"prediction": keras_model.output})

    builder = saved_model_builder.SavedModelBuilder(out_model)
    legacy_init_op = tf.group(
        tf.tables_initializer(), name='legacy_init_op')

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer())

    sess.run(init_op)

    # Add the meta_graph and the variables to the builder
    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                prediction_signature,
        },
        legacy_init_op=legacy_init_op)

    # save the graph
    builder.save()


if __name__ == '__main__':
    # An argument parser for the program
    parser = argparse.ArgumentParser(description="Keras2TF")
    parser.add_argument("--in-model", type=str,
                        default="./MovementData/Walking_02.txt",
                        help="Keras model path")
    parser.add_argument("--out-model", type=str,
                        default="./saved-model",
                        help="TF model dir")
    args = parser.parse_args()

    keras2tf(**vars(args))
