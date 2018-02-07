import numpy as np
import h5py
import argparse
import os
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='lowpass filter for smoothing steering value')
	parser.add_argument('--dataset', type=str, default="bags/test.bag", help='Dataset/ROS Bag name')
	parser.add_argument('--verbose', type=bool, default=False, help='Verbose output')
	args = parser.parse_args()

	dataset = args.dataset
	verbose = args.verbose
	dataset_filename, _ = os.path.splitext(dataset)

	print('Reading HDF5 ', dataset)
	hf = h5py.File(dataset, 'r')

	# Write new HDF5 file
	output_filename = dataset_filename + '-smooth.h5'
	print('Writing HDF5', output_filename)
	mhf = h5py.File(output_filename, 'w')

	hf.keys()

	idx=0

	video = hf.get('video')
	command = hf.get('command')

	image = video.get('image')
	img_stamp = video.get('timestamp')

	steer = command.get('steering')
	throttle = command.get('throttle')
	gear = command.get('gear_shift')
	cmd_stamp = command.get('timestamp')

	#print(img_stamp[:15])
	#print(cmd_stamp[:15])
	# print(gear[:])

	interpol_steer = np.zeros(len(image))
	interpol_throttle = np.zeros(len(image))
	interpol_gear = np.zeros(len(image))

	for idx in range(len(img_stamp)):
		#img_stamp[idx]

		mapidx = np.where(img_stamp[idx] > cmd_stamp)

		if len(mapidx[0]) == 0:
			#print(idx)
			#interpol_steer.append(steer[0])
			interpol_steer[idx] = steer[0]
			interpol_throttle[idx] = throttle[0]
			interpol_gear[idx] = gear[0]
			#print("empty")
		else:
			#print(idx)
			#print(steer[mapidx[0][-1]])
			#print(steer[mapidx[0][-1] + 1])
			average = (steer[mapidx[0][-1]] + steer[mapidx[0][-1] + 1])/2
			interpol_steer[idx] = (average)
			average = (throttle[mapidx[0][-1]] + throttle[mapidx[0][-1] + 1])/2
			interpol_throttle[idx] = (average)
			average = (gear[mapidx[0][-1]] + gear[mapidx[0][-1] + 1])/2
			interpol_gear[idx] = (average)
			#print(average)
			#print(mapidx[0][-1])

	g1 = mhf.create_group('video')
	g1.create_dataset('timestamp',data=img_stamp)
	g1.create_dataset('image',data=image)
	g2 = mhf.create_group('command')
	g2.create_dataset('timestamp',data=img_stamp)
	g2.create_dataset('steering',data=interpol_steer)
	g2.create_dataset('throttle',data=interpol_throttle)
	g2.create_dataset('gear_shift',data=interpol_gear)

	filtered = lowess(interpol_steer, img_stamp, is_sorted=True, frac=0.065, it=0)
	g2.create_dataset('smooth_steering',data=filtered[:,1])
	#print(filtered[:,0])
	#print(filtered[:,1])

	if verbose:
		group2 = mhf.get('command')
		print(group2.items())
		f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)
		ax1.plot(cmd_stamp,steer,linestyle="-")
		ax1.set_title('Sharing both axes')
		ax2.plot(img_stamp,interpol_steer, color='b')
		ax2.plot(filtered[:,0], filtered[:,1], color='r')
		ax3.plot(cmd_stamp,throttle, color='r')
		ax4.plot(img_stamp,interpol_throttle, color='r')
		f.subplots_adjust(hspace=0)
		plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
		plt.show()

	hf.close()
