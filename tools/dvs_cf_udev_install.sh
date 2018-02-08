#!/bin/bash

# This script is taken from RPG ROS package at
# https://github.com/uzh-rpg/rpg_dvs_ros
echo "Copying udev rule (needs root privileges)."
sudo cp 65-inilabs.rules /etc/udev/rules.d/

echo "Copying udev rules for crazyflie"
sudo usermod -a -G plugdev $(whoami)
sudo cp 99-crazyflie.rules /etc/udev/rules.d/
sudo cp 99-crazyradio.rules /etc/udev/rules.d/

echo "Reloading udev rules."
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "Done!"
