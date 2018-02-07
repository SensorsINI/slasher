"""Testing basic utilities of rosbag.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function
import argparse
import os

import numpy as np
import rosbag
import h5py

import spiker
from spiker import log
from spiker.data import rosbag as rb

logger = log.get_logger("rosbag-exporter", log.INFO)


def rosbag2hdf5(data_name):
    """ROS Bag to HDF5 raw format.

    The result HDF5 has all the events and APS frames.
    """
    bag_path = os.path.join(
        spiker.SPIKER_DATA, "rosbag", data_name)
    hdf5_path = bag_path[:-4]+".hdf5"

    bag = rosbag.Bag(bag_path, "r")

    bag_topics = rb.get_topics(bag)

    for key, value in bag_topics.iteritems():
        logger.info("Topic: %s" % (key))

    num_images = rb.get_msg_count(bag, "/dvs/image_raw")
    logger.info("Number of images: %d" % (num_images))
    num_event_pkgs = rb.get_msg_count(bag, "/dvs/events")
    logger.info("Number of event packets: %d" % (num_event_pkgs))
    num_imu_pkgs = rb.get_msg_count(bag, "/dvs/imu")
    logger.info("Number of IMU packets: %d" % (num_imu_pkgs))
    num_pwm_pkgs = rb.get_msg_count(bag, "/raw_pwm")
    logger.info("Number of pwm packets: %d" % (num_pwm_pkgs))
    num_bind_pkgs = rb.get_msg_count(bag, "/dvs_bind")
    logger.info("Number of DVS binded frame packets: %d" % (num_bind_pkgs))
    start_time = bag.get_start_time()  # time in second
    logger.info("Start time: %f" % (start_time))
    end_time = bag.get_end_time()
    logger.info("End time: %f" % (end_time))
    logger.info("Duration: %f s" % (end_time-start_time))

    # image shape
    img_shape = (180, 240)

    # Define HDF5
    dataset = h5py.File(hdf5_path, "w")
    aps_group = dataset.create_group("aps")
    dvs_group = dataset.create_group("dvs")
    imu_group = dataset.create_group("imu")
    extra_group = dataset.create_group("extra")
    pwm_group = extra_group.create_group("pwm")
    bind_group = extra_group.create_group("bind")

    # define dataset
    # APS Frame data
    aps_frame_ds = aps_group.create_dataset(
        name="aps_data",
        shape=(num_images,)+img_shape,
        dtype="uint8")
    aps_time_ds = aps_group.create_dataset(
        name="aps_ts",
        shape=(num_images,),
        dtype="int64")

    # PWM data
    pwm_data_ds = pwm_group.create_dataset(
        name="pwm_data",
        shape=(num_pwm_pkgs, 3),
        dtype="float32")
    pwm_time_ds = pwm_group.create_dataset(
        name="pwm_ts",
        shape=(num_pwm_pkgs,),
        dtype="int64")

    # DVS binded frames data
    bind_data_ds = bind_group.create_dataset(
        name="bind_data",
        shape=(num_bind_pkgs,)+img_shape+(2,),
        dtype="uint8")
    bind_time_ds = bind_group.create_dataset(
        name="bind_ts",
        shape=(num_bind_pkgs,),
        dtype="int64")

    # DVS data
    dvs_data_ds = dvs_group.create_dataset(
        name="event_loc",
        shape=(0, 2),
        maxshape=(None, 2),
        dtype="uint16")
    dvs_time_ds = dvs_group.create_dataset(
        name="event_ts",
        shape=(0,),
        maxshape=(None,),
        dtype="int64")
    dvs_pol_ds = dvs_group.create_dataset(
        name="event_pol",
        shape=(0,),
        maxshape=(None,),
        dtype="bool")
    dvs_packet_time = dvs_group.create_dataset(
        name="packet_ts",
        shape=(num_event_pkgs, 2),
        dtype="int64")

    # topic list
    topics_list = ["/dvs/image_raw/", "/dvs/events", "/dvs/imu",
                   "/raw_pwm", "/dvs_bind"]

    frame_idx = 0
    bind_frame_idx = 0
    pwm_idx = 0
    event_packet_idx = 0
    for topic, msg, t in bag.read_messages(topics=topics_list):
        if topic in ["/dvs/image_raw/"]:
            image = rb.get_image(msg)

            aps_frame_ds[frame_idx] = image
            aps_time_ds[frame_idx] = msg.header.stamp.to_nsec()//1000
            logger.info("Processed %d/%d Frame"
                        % (frame_idx, num_images))
            frame_idx += 1
        elif topic in ["/raw_pwm"]:
            steering = msg.steering
            throttle = msg.throttle
            gear_shift = msg.gear_shift
            time_stamp = t.to_nsec()//1000

            pwm_data_ds[pwm_idx] = np.array([steering, throttle, gear_shift])
            pwm_time_ds[pwm_idx] = time_stamp
            logger.info("Processed %d/%d PWM packet"
                        % (pwm_idx, num_pwm_pkgs))
            pwm_idx += 1
        elif topic in ["/dvs_bind"]:
            image = rb.get_image(msg, "bgr8")[..., :2]

            bind_data_ds[bind_frame_idx] = image
            bind_time_ds[bind_frame_idx] = msg.header.stamp.to_nsec()//1000

            logger.info("Processed %d/%d Frame"
                        % (bind_frame_idx, num_bind_pkgs))
            bind_frame_idx += 1
        elif topic in ["/dvs/events"]:
            events = msg.events
            num_events = len(events)

            # export events
            # record packet time
            dvs_packet_time[event_packet_idx, 0] = \
                msg.header.stamp.to_nsec()//1000
            # record packet starting index
            dvs_packet_time[event_packet_idx, 1] = dvs_data_ds.shape[0]
            # record events
            events_loc_arr = np.array([[x.x, x.y] for x in events],
                                      dtype=np.uint16)
            events_ts_arr = np.array([x.ts.to_nsec()//1000 for x in events],
                                     dtype=np.int64)
            events_pol_arr = np.array([x.polarity for x in events],
                                      dtype=np.bool)

            resized_shape = dvs_data_ds.shape[0]+num_events

            dvs_data_ds.resize(resized_shape, axis=0)
            dvs_time_ds.resize(resized_shape, axis=0)
            dvs_pol_ds.resize(resized_shape, axis=0)

            dvs_data_ds[-num_events:] = events_loc_arr
            dvs_time_ds[-num_events:] = events_ts_arr
            dvs_pol_ds[-num_events:] = events_pol_arr

            logger.info("Processed %d/%d event packet, Seq: %d"
                        % (event_packet_idx, num_event_pkgs,
                           msg.header.seq))
            event_packet_idx += 1

    dataset.close()


if __name__ == '__main__':
    # An argument parser for the program
    parser = argparse.ArgumentParser(description="ROS bag to HDF5 raw format")
    parser.add_argument("--data-name", "-n", type=str,
                        default="",
                        help="name of your dataset in spikeres folder")
    args = parser.parse_args()
    rosbag2hdf5(**vars(args))
