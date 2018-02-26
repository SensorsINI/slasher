"""Resave a Keras model into Tensorflow.

This script is a restructured version of
https://github.com/amir-abdi/keras_to_tensorflow

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import argparse

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import keras.backend as K
from keras.backend import tf as ktf
from keras.models import load_model
from keras.models import model_from_json


def keras2tf(in_model, in_model_mode, out_model):
    """Convert Keras model to TensorFlow model."""
    # Load Keras Model
    sess = tf.Session()
    K.set_session(sess)
    K.set_learning_phase(0)

    if in_model_mode == 0:
        keras_model = load_model(in_model)
    else:
        with open(in_model, 'r') as json_file:
            json_model = json_file.read()
            keras_model = model_from_json(
                json_model, custom_objects={"ktf": ktf})
        print('Pilot model is loaded...')
        pre_trained_weights = in_model.replace('json', 'h5')
        keras_model.load_weights(pre_trained_weights)

    num_output = 1
    pred = [None]*num_output
    pred_node_names = [None]*num_output
    for i in range(num_output):
        pred_node_names[i] = "prediction"+str(i)
        pred[i] = tf.identity(
            keras_model.outputs[i], name=pred_node_names[i])
    print('output nodes names are: ', pred_node_names)

    constant_graph = graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(
        constant_graph, ".", out_model, as_text=False)
    print('saved the freezed graph (ready for inference) at: ',
          out_model)


if __name__ == '__main__':
    # An argument parser for the program
    parser = argparse.ArgumentParser(description="Keras2TF")
    parser.add_argument("--in-model", type=str,
                        default="",
                        help="Keras model path")
    parser.add_argument("--in-model-mode", type=int,
                        default="",
                        help="Keras model path")
    parser.add_argument("--out-model", type=str,
                        default="./saved-model",
                        help="TF model dir")
    args = parser.parse_args()

    keras2tf(**vars(args))
