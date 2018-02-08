"""Compare Steering.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import os

import matplotlib.pyplot as plt

import numpy as np
import rosbag

import spiker

#  data_name = "compare_2018-02-07-22-39-11.bag"
#  data_name = "compare_2018-02-07-22-15-17.bag"
#  data_name = "compare_new_2018-02-07-23-00-10.bag"
data_name = "compare_2018-02-08-15-04-51.bag"

bag_path = os.path.join(
    spiker.SPIKER_DATA, "rosbag", data_name)
hdf5_path = bag_path[:-4]+".hdf5"

bag = rosbag.Bag(bag_path, "r")

topics_list = ["/raw_pwm", "/drive_pwm"]

frame_idx = 0
bind_frame_idx = 0
pwm_idx = 0
event_packet_idx = 0

pwm_value_list = []
pwm_time_list = []
drive_value_list = []
drive_time_list = []
for topic, msg, t in bag.read_messages(topics=topics_list):
    if topic in ["/raw_pwm"]:
        time_stamp = t.to_nsec()//1000

        pwm_value_list.append(msg.steering)
        pwm_time_list.append(time_stamp)
    elif topic in ["/drive_pwm"]:
        time_stamp = t.to_nsec()//1000

        drive_value_list.append(msg.steering)
        drive_time_list.append(time_stamp)

time_diff = drive_time_list[0]
drive_time_list = np.array(drive_time_list)-time_diff
pwm_time_list = np.array(pwm_time_list)-time_diff

plt.figure()
plt.plot(pwm_time_list/1e6, pwm_value_list, "r",
         drive_time_list/1e6-3.5,
         np.array(drive_value_list)*500+1500, "g")
plt.show()
