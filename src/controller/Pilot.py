#!/usr/bin/env python
"""Autopilot mode.

This script activate pilot mode to take control over Jetson Car.

Use X button on joystick to stop
"""
from __future__ import print_function

import threading

import rospy
from cv_bridge import CvBridge
import tensorflow as tf

from sensor_msgs.msg import Joy, Image
from rally_msgs.msg import Pwm

steering = 0.0
throttle = 0.0
cv_bridge = CvBridge()

print("Building Pilot Model...")


class Pilot:
    # Activate autonomous mode in Jetson Car
    def __init__(self, get_model_call_back, model_callback,
                 img_proc_callback, img_config=None):
        global graph
        self.image = None
        self.model = None
        self.event_img = None

        # get model
        self.get_model = get_model_call_back
        self.model = self.get_model()
        self.model._make_predict_function()
        graph = tf.get_default_graph()

        self.predict = model_callback
        self.img_proc = img_proc_callback
        self.completed_cycle = False
        self.start = 0.
        self.lock = threading.RLock()
        self.img_config = img_config
        self.img_shape = img_config["img_shape"]
        self.histrange = [(0, v) for v in self.img_shape]
        self.clip_value = img_config["clip_value"]
        self.mode = img_config["mode"]

        # Load Keras Model - Publish topic - CarController
        rospy.init_node("pilot_steering_model", anonymous=True)
        # load keras model from start
        self.joy = rospy.Subscriber('joy', Joy, self.joy_callback)
        self.control_signal = rospy.Publisher('/drive_pwm', Pwm, queue_size=1)

        # subscriber for image and event
        if self.mode in [0, 2]:
            self.camera = rospy.Subscriber(
                '/dvs_bind', Image, self.callback, queue_size=1,
                buff_size=262144)
        else:
            self.camera = rospy.Subscriber(
                '/dvs/image_raw', Image, self.callback, queue_size=1,
                buff_size=262144)

        # Lock which waiting for Keras model to make prediction
        # why?
        rospy.Timer(rospy.Duration(0.005), self.send_control)

    def joy_callback(self, joy):
        global throttle
        throttle = joy.axes[3]  # Let user can manual throttle

    def callback(self, camera_info):
        global steering, throttle, graph
        if self.lock.acquire(True):
            # get aps image
            self.image = cv_bridge.imgmsg_to_cv2(
                camera_info, "bgr8")[..., :2] if self.mode == 2 else \
                cv_bridge.imgmsg_to_cv2(camera_info, "mono8")

            # do custom image processing here
            input_img = self.img_proc(self.image,
                                      config=self.img_config)

            with graph.as_default():
                steering, _ = self.predict(self.model, input_img)
            self.completed_cycle = True
            self.lock.release()

    def send_control(self, event, verbose=False):
        global steering, throttle
        if self.image is None:
            return
        if self.completed_cycle is False:
            return
        # Publish a rc_car_msgs
        msg = Pwm()
        msg.steering = steering
        msg.throttle = throttle
        self.control_signal.publish(msg)
        if verbose is True:
            print("Steering: %.2f. Throttle: %.2f" % (steering, throttle))
        self.completed_cycle = False
