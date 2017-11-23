import os
import argparse
import rosbag
from cv_bridge import CvBridge, CvBridgeError
import h5py
import progressbar
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='rosbag to HDF5 converter for RPG DAVIS driver')
    parser.add_argument('--dataset', type=str, default="bags/Testing.bag", help='Dataset/ROS Bag name')
    parser.add_argument('--verbose', type=bool, default=False, help='Verbose output')
    args = parser.parse_args()
 
    dataset = args.dataset
    verbose = args.verbose
 
    dataset_filename, _ = os.path.splitext(dataset)
 
    output_x = dataset_filename + '.h5'
 
    first_frame_read = False
    first_command_read = False

    last_frame = None
    last_angle = None

    instance = 0

    image_count = 0
    command_count = 0
 
    bridge = CvBridge()
 
    x_file = h5py.File(output_x, 'w')
    x_dset = None
    y_dset = None
 
    print('reading rosbag ', dataset)
    bag = rosbag.Bag(dataset, 'r')
 
    image_instances = bag.get_message_count('/dvs/image_raw')
    command_instances = bag.get_message_count('/raw_pwm')
    total_instances = image_instances + command_instances
    #skipped_instances = 0
 
    progress = progressbar.ProgressBar(maxval=total_instances)
 
    progress.start()
 
    for topic, msg, t in bag.read_messages(topics=['/dvs/image_raw', '/raw_pwm',]):
        if verbose:
            if topic in ['/dvs/image_raw',]:
                print(topic, msg.header.seq, t-msg.header.stamp, msg.height, msg.width, msg.encoding, t)
            else:
                # Angles from -8.2 to 8.2 rad since a steering wheel can do more than one rotation
                print(topic, msg.header.seq, t-msg.header.stamp, msg.steering_wheel_angle, t)
 
        if topic in ['/dvs/image_raw',]:
            if not first_frame_read:
                first_frame_read = True
                im_gray = bridge.imgmsg_to_cv2(msg, "mono8") #mono8

                g1 = x_file.create_group('video')
                i_timestamp = g1.create_dataset('timestamp', (image_instances, ), dtype='int64')
                images = g1.create_dataset('image',(image_instances, im_gray.shape[0], im_gray.shape[1], im_gray.shape[2]), dtype='uint8')
            try:
                im_gray = bridge.imgmsg_to_cv2(msg, "mono8") #mono8
                images[image_count] = im_gray
                i_timestamp[image_count] = msg.header.stamp.to_nsec()
                image_count += 1
            except CvBridgeError as e:
                 print(e)

        if topic in ['/raw_pwm',]:
            if not first_command_read:
                first_command_read = True
                g2 = x_file.create_group('command')
                c_timestamp = g2.create_dataset('timestamp', (command_instances, ), dtype='int64')
                steering = g2.create_dataset('steering', (command_instances, ), dtype='float')
                throttle = g2.create_dataset('throttle', (command_instances, ), dtype='float')
                gear_shift = g2.create_dataset('gear_shift', (command_instances,  ), dtype='float')
            try:
                c_timestamp[command_count] = t.to_nsec()
                steering[command_count] = msg.steering
                throttle[command_count] = msg.throttle
                gear_shift[command_count] = msg.gear_shift
                command_count += 1

                progress.update(image_count+command_count)
                #print(instance)
            except:
                print("ha")
 
    progress.finish()