# monstruck
A aggressive self-driving stadium truck.

## Install ROS

1. Install ROS for host machine

The installation script is at [here](./tools/host_setup.sh)

```
sudo ./host_setup.sh
```

2. Install ROS for ODROID UX4/Jetson TX2

The installation script is at [here](./tools/car_setup.sh)

```
sudo ./car_setup.sh
```

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
$ echo "source $HOME/catkin_ws/devel/setup.sh" >> ~/.bashrc  # please modify accordingly if you use other shells
$ source ~/.bashrc
```

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
sudo ./dvs_cf_udev_install.sh
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

Configure bias parameters to DVS camera, see DVS calibration for more details.

Configure autonomous driving parameters by modifying `param_server.yaml`.

1. Launch the main controller on the car

```
$ roslaunch controller manual.launch
```

Launch crazyflie node on the host machine

```
$ roslaunch controller crazyflie.launch uri:=radio://0/yourchannel/yourfreq
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

1. Set six LSP nodes to pre-defined locations, connect the nodes to USB power banks.

2. Modify `anchor_pos.yaml`

```
$ cd ~/catkin_ws/src/lps-ros/data
$ mv anchor_pos.yaml.sample anchor_pos.yaml  # only do once
```

Set each anchor position in following template

```
anchor[id]_pos: [X_pos, Y_pos, Z_pos]
```

3. Find available radio URI configuration

```
$ rosrun crazyflie_tools scan
```

4. Launch calibration node

```
$ roslaunch controller calibration_crazyflie_host.launch uri:=radio://0/yourchannel/yourfreq
```

## Prepare training for monstruck data

We provided a set of tools in [tools](./tools) and [model](./model) folders so that you can use to manage rosbag recordings, prepare training data, and build training models.

### Requirements

To use these tools, there are several requirements

+ TensorFlow
+ Keras
+ scikit-image
+ numpy
+ scipy
+ spiker (a private repository, ask for access)
+ rospy
+ rosbag
+ sacred

### Convert ROS bag to HDF5 format

[This script](./tools/rosbag_exporter.py) converts a ROS bag recording to HDF5 format.
The output HDF5 file is structured as a valid HDF5 recording file and can be accessed easily through `h5py`. The usage is

```
$ python rosbag_exporter -n your_rosbag_recording.bag
```

Note that your recording is in your `spikeres/data/rosbag` directory.
Please read the code for more details.

### Export HDF5 recordings to ready-to-train HDF5 datasets

The raw HDF5 recording provides all data, however, to train a Neural Network,
you won't need everything. Therefore, you can prepare the training and testing dataset through [hdf5_exporter.py](./tools/hdf5_exporter.py). Usage:

```
$ python hdf5_exporter.py -n your_hdf5_recording.hdf5 -c /path/to/preprocessing_config.json
```

The current script receives a preprocessing config JSON file that is defined by the training experiment (see below, we provided a example of the current JSON file as well) in order to match the exact training condition.
The raw HDF5 recording is in `spikeres/data/rosbag` and the preprocessing config file can be anywhere in the system.

### Joining multiple training dataset together

[hdf_binder.py](./tools/hdf_binder.py) is a tool that binds multiple HDF5 datasets together into a single HDF5 document. Usage:

```
$ python hdf_binder.py -n file_list.json
```

The `file_list.json` has three fields as the example showed below:

```json
{
    "file_list": ["hdf5_dataset_1.hdf5",
                  "hdf5_dataset_2.hdf5"],
    "bind_name": "binded_hdf5.hdf5",
    "img_shape": [30, 90]
}
```

### Training with ResNet

[The model training script](./model/monstruck_drive_model.py) is to train a
ResNet model with the training and testing datasets.

We use `sacred` to configure each experiments. The usage of this script is:

```
$ python monstruck_drive_model.py with configs/monstruck_drive_model_exp.json
```

`sacred` can serialize your experiment configurations so that you can repeat different experiments with a single script. A example of this configuration file is at [here](./model/configs/monstruck_drive_model_exp.json)

### Evaluating and exporting the model file for monstruck controller

[The evaluation script](./model/resnet_model_inference.py) loads a trained model
and then evaluates on a test dataset. It also re-saves the trained model to model definition JSON file and weights file in HDF5. These two files are used by monstruck to perform inference. Please check out the script and modify to your needs.

## Contacts

Hong Ming Chen, Yuhuang Hu
