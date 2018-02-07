# monstruck
A aggressive self-driving stadium truck.

## Install ROS

1. Install ROS for Desktop

[TODO] setup script in bash

2. Install ROS for ODROID UX4/Jetson TX2

[TODO] setup script in bash

[TODO] make sure ros's setup script to bashrc

## Install DVS and related drivers

1. Install necessary ROS packages and tools

```
$ sudo apt-get install libusb-1.0-0-dev
$ sudo apt-get install ros-kinetic-camera-info-manager
$ sudo apt-get install ros-kinetic-image-view
$ sudo apt-get install ros-kinetic-cv-bridge
$ sudo apt-get install gcc-avr arduino arduino-core
$ sudo apt-get install ros-kinetic-rosserial
$ sudo apt-get install ros-kinetic-rosserial-arduino
$ sudo apt-get install ros-kinetic-angles
$ sudo apt-get install ros-kinetic-joy
$ sudo apt-get install v4l-utils  # maybe optional 
$ sudo apt-get install python-catkin-tools
```

Install Python related packages on ODROID UX4/Jetson TX2

```
$ sudo pip install future -U
$ sudo pip install numpy -U
$ sudo pip install scipy -U
$ sudo pip install scikit-image -U
$ sudo pip install tensorflow -U
$ sudo pip install keras -U
```

2. Create catkin workspace

```
$ cd
$ mkdir -p catkin_ws/src
$ cd catkin_ws
$ catkin config --init --mkdirs --extend /opt/ros/kinetic --merge-devel --cmake-args -DCMAKE_BUILD_TYPE=Release
```

[TODO] add catkin setup to bashrc

3. Build RPG DVS drivers and `monstruck`

__NOTE__: we use our customized `rpg_dvs_ros` drivers, please follow the instruction [here](https://github.com/uzh-rpg/rpg_dvs_ros) if you want to use the original project.

```
$ cd ~/catkin_ws/src
$ git clone https://github.com/catkin/catkin_simple.git
$ git clone https://github.com/duguyue100/rpg_dvs_ros
$ git clone https://github.com/NeuromorphicProcessorProject/monstruck
$ git clone https://github.com/NeuromorphicProcessorProject/crazyflie_ros 
$ git clone https://github.com/NeuromorphicProcessorProject/lps-ros
$ cd ..
$ catkin build
$ source ~/.bashrc
```

Update `udev` rules for DVS cameras and crazyflie

```
[TODO] write a bash file
```

__NOTE__: you will have to `source ~/.bashrc` after every build.

__NOTE__: If you want to clean-build packages, please follow:

```
$ cd ~/catkin_ws
$ catkin clean
$ catkin build
```

## Usage

0. Configure related parameters for the main controller

Configure bias parameters to DVS camera

```
$ cp ~/catkin_ws/src/rpg_dvs_ros/davis_ros_driver/config/DAVIS240C.yaml ~/.ros/camera_info
$ mv DAVIS240C.yaml DAVIS240-024600xx.yaml
```

[TODO] maybe serialize davis bias here, maybe use `rqt_reconfigure`

Configure autonomous driving parameters by modifying `param_server.yaml`.

1. Launch the main controller

```
$ roslaunch controller manual.launch
```

2. Joystick control command

+ `Y`: take control back from hand held controller
+ `B`: firmware level emergency brake
+ `A`: start recording
+ `X`: stop recording
+ `LB`: joystick level emergency brake
+ `LT`: start joystick manual control
+ `RB`: stop autonomous control
+ `RT`: start autonomous control

+ Left analog mini-stick: steering (rotate left and right)
+ Right analog mini-stick: throttle (rotate up and down)

## Calibration and setup for DVS camera and crazyflie

### DVS camera calibration

To start DVS camera calibration on the car

```
roslaunch controller calibration_car.launch
```

To start DVS camera calibration on the host machine

```
roslaunch controller calibration_host.launch
```

You can calibrate camera configs from the `rqt_reconfigure` GUI tool.
Then you will need to save the change to `monstruck_camera_config.yaml`
The file is located at

```
$ ~/catkin_ws/src/monstruck/src/controller/cfg
```

### Crazyflie calibration

1. Set six LSP node to pre-defined location, connect the nodes to USB power banks.

2. Modify `anchor_pos.yaml`

```
$ cd ~/catkin_ws/src/lps-ros/data
$ mv anchor_pos.yaml.sample anchor_pos.yaml  # only do once
```

Set each anchor position based on

```
anchor[id]_pos: [X_pos, Y_pos, Z_pos]
```

3. Find available radio URI configuration

```
$ rosrun crazyflie_tools scan
```

4. Launch calibration node

```
$ roslaunch controller calibration_crazyflie_host.launch
```

## Contacts

Hong Ming Chen, Yuhuang Hu
