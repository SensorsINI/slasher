#!/usr/bin/env python
"""Create Pilot Model."""
import os
import rospy
import numpy as np
from skimage.transform import resize
from keras.models import model_from_json
from keras.backend import tf as ktf
from Pilot import Pilot


def drive(model, image):
    """ Make prediction on steering angle given an image

    # Parameters
    model : A valid Keras model
    image : numpy.ndarray
        A input image

    # Returns
    steering_angle : float
        steering angle to send
    throttle : float
        throttle value to send
    """
    if image is None:
        return

    # predict output
    prediction = model.predict(image)
    steering_angle = prediction[0][0]
    throttle = 0.1

    return steering_angle, throttle


def load_model(model_path):
    """Load a Keras model.

    # Parameters
    model_path : str
        absolute path to the Keras model.

    # Returns
    model : A Keras model.
    """
    with open(model_path, 'r') as json_file:
        json_model = json_file.read()
        model = model_from_json(json_model, custom_objects={"ktf": ktf})
    print('Pilot model is loaded...')
    pre_trained_weights = model_path.replace('json', 'h5')
    model.load_weights(pre_trained_weights)

    return model


def img_preproc(in_image, config=None):
    """Do custom image preprocssing here.

    # Parameters
    in_image : numpy.ndarray
        input image, could be one or two channels
    config : dict
        dictionary that contains configuration
        for the input image of the target model

    # Returns
    img : numpy.ndarray
        a 4-D tensor that is a valid input to the model.
    """
    # return image if there is no image
    if in_image is None:
        return in_image

    # load config
    if config is not None:
        frame_cut = config["frame_cut"]
        target_size = config["target_size"]
        mode = config["mode"]
        clip_value = 255. if mode == 1 else float(config["clip_value"])*2
    else:
        mode = 2  # 0: DVS, 1: APS, 2: combined

    # append axis if no axis
    in_image = in_image[..., np.newaxis] if mode in [0, 1] else in_image

    # cut useless content
    in_image = in_image[frame_cut[0][0]:-frame_cut[0][1],
                        frame_cut[1][0]:-frame_cut[1][1], :].astype("float32")

    # re-normalize image
    if mode in [0, 1]:
        in_image /= clip_value
    else:
        # DVS
        in_image[..., 0] /= clip_value
        # APS
        in_image[..., 1] /= 255.

    # resize to target size
    in_image = resize(in_image, target_size, mode="reflect")

    return in_image[np.newaxis, ...].astype(np.float32)


if __name__ == "__main__":
    package_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        package_path, "..", "..", "pretrained_models",
        rospy.get_param('model_path'))
    print("Activating AutoPilot model..\n")
    img_config = {}
    img_config["frame_cut"] = rospy.get_param("frame_cut")
    img_config["img_shape"] = tuple(rospy.get_param("img_shape"))
    img_config["target_size"] = tuple(rospy.get_param("target_size"))
    img_config["clip_value"] = rospy.get_param("clip_value")
    img_config["mode"] = rospy.get_param("mode")
    print (img_config)
    pilot = Pilot(lambda: load_model(model_path), drive,
                  img_preproc, img_config=img_config)
    rospy.spin()
