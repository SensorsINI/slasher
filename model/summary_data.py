"""Summary data.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""
from __future__ import print_function, absolute_import
import os

import numpy as np
import random
import h5py

import matplotlib.pyplot as plt

import spiker


def data_balance_gen(Y_train, batch_size=128):
    while True:
        steerings = np.zeros((batch_size,), dtype=np.float32)
        for i in range(batch_size):
            straight_count = 0
            for i in range(batch_size):
                # Select random index to use for data sample
                sample_index = random.randrange(len(Y_train))

                angle = Y_train[sample_index]
                if abs(angle) < .1:
                    straight_count += 1
                if straight_count > (batch_size * .2):
                    while abs(Y_train[sample_index]) < .1:
                        sample_index = random.randrange(len(Y_train))
                        angle = Y_train[sample_index]
                steerings[i] = angle

        yield steerings


# import train data
train_path = os.path.join(
    spiker.HOME, "data", "exps", "data", "jogging-train.hdf5")
test_path = os.path.join(
    spiker.HOME, "data", "exps", "data", "jogging-test.hdf5")
#  train_path = os.path.join(
#      spiker.HOME, "data", "exps", "data",
#      "foyer-train.hdf5")
#  test_path = os.path.join(
#      spiker.HOME, "data", "exps", "data",
#      "foyer-test.hdf5")


train_data = h5py.File(train_path, "r")
test_data = h5py.File(test_path, "r")

train_pwm = train_data["pwm"][()]
test_pwm = test_data["pwm"][()]
pwm = np.append(train_pwm, test_pwm, axis=0)
#  pwm = train_pwm

train_data.close()
test_data.close()

# throttle
#  throttle = (pwm[:, 1]-1000)/1000
#  throttle_up = np.percentile(throttle, 75)
#  throttle_down = np.percentile(throttle, 25)
#  IQR = throttle_up-throttle_down
#  throttle_up += 1.5*IQR
#  throttle_down -= 1.5*IQR

#  print (throttle_down)
#  th_up_index = (throttle < throttle_up)
#  throttle = throttle[th_up_index]
#  th_down_index = (throttle > throttle_down)
#  throttle = throttle[th_down_index]
#  print (throttle.shape)

# Steering
steering = pwm[:, 0]
#  steering = steering[th_up_index]
#  steering = steering[th_down_index]

#  plt.figure()
#  plt.hist(steering*25, bins=51, facecolor='g', alpha=0.75,
#           edgecolor='black', linewidth=1.2)
#  plt.xlabel("steering angle (degree)", fontsize=16)
#  plt.ylabel("number of instance", fontsize=16)
#  plt.xticks(fontsize=16)
#  plt.yticks(fontsize=16)
#  plt.grid()
#  plt.plot(steering)
#  plt.boxplot(steering)
#  plt.show()

# balance data
steering_balanced = np.array([])
steer = data_balance_gen(steering, batch_size=128)
for batch in xrange(steering.shape[0]//128+1):
    steering_balanced = np.append(
        steering_balanced, steer.next())
    print ("batch %d/%d" % (batch+1, steering.shape[0]//128))


plt.figure()
plt.hist([steering*25, steering_balanced*25], bins=26,
         alpha=0.8,
         edgecolor='black', linewidth=1.2,
         label=["original", "balanced"])
#  plt.hist(steering_balanced*25, bins=51, facecolor='g', alpha=0.5,
#           edgecolor='black', linewidth=1.2)
plt.xlabel("steering angle (degree)", fontsize=16)
plt.ylabel("number of instance", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.legend(fontsize=16)
plt.show()
#
#  plt.figure()
#  plt.plot(np.array(range(steering.shape[0])), steering, alpha=0.5, color="r")
#  plt.xticks(fontsize=16)
#  plt.yticks(fontsize=16)
#  plt.grid()
#  plt.show()
