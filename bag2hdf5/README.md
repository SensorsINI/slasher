# ROSBAG to HDF5
## Commandline API Usage
### Convert specicific ROSBAG file and store in the same place
```sh=
python bag2hdf5.py --dataset Path_to_Dataset/ROS_Bag_name
```
### Print out '/dvs/image_raw' & '/raw_pwm' data
```sh=
python bag2hdf5.py --verbose
```

# Interploation and Smooth Steering Dataset
1. Interploate steering angle with frame timestamp.
2. Apply lowpass filter to smooth steering angle value.
## Commandline API Usage
### Convert specicific ROSBAG file and store in the same place
```sh=
python smooth_steering.py --dataset Path_to_Dataset/HDF5_name
```
### Print figure of original steering angle and interploated steering angle with or without smoothed result.
```sh=
python smooth_steering.py --verbose
```