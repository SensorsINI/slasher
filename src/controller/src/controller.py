#!/usr/bin/env python
"""Main controller.

This script is used to start/stop record rosbag using Joy ROS message
"""
from __future__ import print_function
import os
import rospy
import std_msgs.msg
import subprocess
import signal
from sensor_msgs.msg import Joy


def callback(joy):
    global autonomous, is_recording, activate_pilot, \
        activate_record

    # RIGHT TOP - BIG BUTTON - RT
    if joy.buttons[7] == 1:
        if autonomous is False:
            autonomous = True
            activate_pilot = subprocess.Popen(
                'roslaunch controller autopilot.launch',
                stdin=subprocess.PIPE, shell=True)
            print("\n\n", '-'*30)
            print("Starting Autonomous Mode.")
            print('-'*30, "\n")
        else:
            print("\n\n", '-'*30)
            print("Autonomous has been activated already.")
            print('-'*30, "\n")

    # RIGHT TOP - SMALL BUTTON - RB
    if joy.buttons[5] == 1:
        if autonomous is True:
            autonomous = False
            terminate_process_and_children(activate_pilot)
            print("\n\n", '-'*30)
            print("Turning off Autonomous...")
            print('-'*30, "\n")
        else:
            print("\n\n", '-'*30)
            print("Autopilot is not activated yet..."
                  "Press RightTop button to activate.")
            print('-'*30, "\n")

    # A - START RECORDING
    if joy.buttons[1] == 1:
        # TODO: see if can read manual mode status
        if autonomous is False and is_recording is False:
            is_recording = True
            recording_path = os.path.join(os.environ["HOME"], "monstruck_rec")
            if not os.path.isdir(recording_path):
                os.makedirs(recording_path)
            activate_record = subprocess.Popen(
                "roslaunch controller recording.launch",
                stdin=subprocess.PIPE, shell=True)
            print("\n\n", '-'*30)
            print("Starting Recording.")
            print('-'*30, "\n")
        else:
            print("\n\n", '-'*30)
            print("Not recording during autonomous mode or during recording.")
            print('-'*30, "\n")

    # X - Stop recording
    if joy.buttons[0] == 1:
        if is_recording is True:
            is_recording = False
            terminate_process_and_children(activate_record)
            print("\n\n", '-'*30)
            print("Turning off Recording...")
            print('-'*30, "\n")
        else:
            print("\n\n", '-'*30)
            print("Nothing to terminated.")
            print('-'*30, "\n")

    # resume control from hand controller - Y
    if joy.buttons[3] == 1:
        resume_control.publish(1)
        print("\n\n", '-'*30)
        print("Joystick in control")
        print('-'*30, "\n")

    if joy.buttons[2] == 1:
        firmware_stop.publish(1)
        print("\n\n", '-'*30)
        print("Firmware level emergency stop")
        print('-'*30, "\n")


def terminate_process_and_children(p):
    ps_command = subprocess.Popen(
        "ps -o pid --ppid %d --noheaders" % p.pid,
        shell=True, stdout=subprocess.PIPE)
    ps_output = ps_command.stdout.read()
    retcode = ps_command.wait()
    assert retcode == 0, "ps command returned %d" % retcode
    for pid_str in ps_output.split("\n")[:-1]:
        os.kill(int(pid_str), signal.SIGINT)
    p.terminate()


def start():
    rospy.init_node("autonomous_controller", anonymous=True)
    rospy.Subscriber("joy", Joy, callback)
    rospy.spin()


if __name__ == "__main__":
    is_recording = False
    autonomous = False
    resume_control = rospy.Publisher(
        "/resumeAuto", std_msgs.msg.Bool, queue_size=1)
    firmware_stop = rospy.Publisher(
        "/eStop", std_msgs.msg.Bool, queue_size=1)
    start()
