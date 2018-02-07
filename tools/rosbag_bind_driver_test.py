"""Testing basic utilities of rosbag.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
from builtins import range
import os

import numpy as np
import rosbag
import cv2

import spiker
from spiker import log
from spiker.data import rosbag as rb

logger = log.get_logger("rosbag-new-bind-test", log.INFO)

bag_path = os.path.join(
    spiker.SPIKER_DATA, "rosbag",
    "monstruck_rec_2018-01-22-19-22-34.bag")
bag = rosbag.Bag(bag_path, "r")

bag_topics = rb.get_topics(bag)

for key, value in bag_topics.iteritems():
    logger.info("Topic: %s" % (key))

num_images = rb.get_msg_count(bag, "/dvs_bind")
logger.info("Number of images: %d" % (num_images))
start_time = bag.get_start_time()  # time in second
logger.info("Start time: %f" % (start_time))
end_time = bag.get_end_time()
logger.info("End time: %f" % (end_time))
logger.info("Duration: %f s" % (end_time-start_time))

# image shape
img_shape = (180, 240)

# topic list
topics_list = ["/dvs_bind"]

frame_idx = 0
pwm_idx = 0
event_packet_idx = 0
for topic, msg, t in bag.read_messages(topics=topics_list):
    if topic in ["/dvs_bind"]:
        image = rb.get_image(msg, encoding="bgr8")

        print(image.shape)

        cv2.imshow("aps", image[..., 1])
        cv2.imshow("dvs", image[..., 0]/16.)
        cv2.waitKey(40)

        frame_idx += 1
