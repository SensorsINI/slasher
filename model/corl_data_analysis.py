"""CORL data analysis and plots.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""
from __future__ import print_function, absolute_import
from builtins import range
import cPickle as pickle
import os

import numpy as np
import matplotlib.pyplot as plt

import spiker


def collect_curves(task, balance, num_trails):
    base_path = os.path.join(
        spiker.HOME, "data", "exps", "models_single")

    # curves collector
    curves_collector = []
    for curves_idx in range(1, num_trails+1):
        # get path
        curve_path = os.path.join(
            base_path, task+"_model_"+balance+"_"+str(curves_idx),
            "csv_history.log")
        print (curve_path)

        # load data
        curve = np.loadtxt(
            curve_path, delimiter=",", skiprows=1)
        curves_collector.append(curve)

    return curves_collector


def get_test_loss_curve(curves):
    num_trails = len(curves)
    num_steps = curves[0].shape[0]

    mean = []
    std = []
    for step_idx in range(num_steps):
        loss_collect = []
        for trail_idx in range(num_trails):
            loss_collect.append(
                curves[trail_idx][step_idx, 6])
        mean.append(np.mean(loss_collect))
        std.append(np.std(loss_collect))
    return np.array(mean), np.array(std)


def get_best_test_steering_throttle(curves):
    num_trails = len(curves)

    min_steering_losses = []
    #  min_throttle_losses = []
    for curve_idx in range(num_trails):
        # get the one with minimum testing loss
        min_loss = np.min(curves[curve_idx][:, 4])
        #  min_steering_loss = curves[curve_idx][min_loss_idx, 8]
        #  min_throttle_loss = curves[curve_idx][min_loss_idx, 10]

        min_steering_losses.append(min_loss)
        #  min_throttle_losses.append(min_throttle_loss)

    return min_steering_losses


#  option = "get-rmse"
#  option = "get-loss-curve"
option = "get-foyer-figure"

if option == "get-rmse":

    # collect curves
    foryer_wo = collect_curves("foryer", "wo_balance", 10)
    foryer_w = collect_curves("foryer", "w_balance", 10)
    jogging_wo = collect_curves("jogging", "wo_balance", 10)
    jogging_w = collect_curves("jogging", "w_balance", 10)

    # Get best testing steering and throttle
    foryer_wo_steer = get_best_test_steering_throttle(foryer_wo)
    foryer_w_steer = get_best_test_steering_throttle(foryer_w)
    jogging_wo_steer = get_best_test_steering_throttle(jogging_wo)
    jogging_w_steer = get_best_test_steering_throttle(jogging_w)

    # get mean and std for steering
    print ("For steering")
    print ("Foryer wo mean: %f, std: %f" % (
        np.sqrt(np.mean(foryer_wo_steer))*25,
        np.sqrt(np.std(foryer_wo_steer))*25))
    print ("Foryer w mean: %f, std: %f" % (
        np.sqrt(np.mean(foryer_w_steer))*25,
        np.sqrt(np.std(foryer_w_steer))*25))

    print ("jogging wo mean: %f, std: %f" % (
        np.sqrt(np.mean(jogging_wo_steer))*25,
        np.sqrt(np.std(jogging_wo_steer))*25))
    print ("jogging w mean: %f, std: %f" % (
        np.sqrt(np.mean(jogging_w_steer))*25,
        np.sqrt(np.std(jogging_w_steer))*25))

    #  print ("For throttle")
    #  print ("Foryer wo mean: %f, std: %f" % (
    #      np.sqrt(np.mean(foryer_wo_throttle))*100,
    #      np.sqrt(np.std(foryer_wo_throttle))*100))
    #  #  print ("Foryer w mean: %f, std: %f" % (
    #  #      np.sqrt(np.mean(foryer_w_throttle))*100,
    #  #      np.sqrt(np.std(foryer_w_throttle))*100))
    #  print ("jogging wo mean: %f, std: %f" % (
    #      np.sqrt(np.mean(jogging_wo_throttle))*100,
    #      np.sqrt(np.std(jogging_wo_throttle))*100))
    #  print ("jogging w mean: %f, std: %f" % (
    #      np.sqrt(np.mean(jogging_w_throttle))*100,
    #      np.sqrt(np.std(jogging_w_throttle))*100))

elif option == "get-foyer-figure":
    base_path = os.path.join(
        spiker.HOME, "data", "exps", "models_single")

    with open(os.path.join(base_path, "jogging-wo-balance.pkl"), "r") as f:
        Y_test, predictions_wo = pickle.load(f)
        f.close()
    with open(os.path.join(base_path, "jogging-w-balance.pkl"), "r") as f:
        Y_test, predictions_w = pickle.load(f)
        f.close()

    Y_mean_wo = np.mean(predictions_wo, axis=0)[:, 0]
    Y_std_wo = np.std(predictions_wo, axis=0)[:, 0]

    Y_mean_w = np.mean(predictions_w, axis=0)[:, 0]
    Y_std_w = np.std(predictions_w, axis=0)[:, 0]

    num_steps = np.array(range(Y_test.shape[0]))/30.

    plt.figure()
    plt.plot(num_steps, Y_test, lw=3,
             label="groundtruth",
             color="#7570b3", ls="-", mew=5,
             alpha=0.75)

    plt.plot(num_steps, Y_mean_wo, lw=2,
             label="predicted (unbalanced)",
             color="#d95f02", ls="-", mew=5)
    plt.fill_between(num_steps, Y_mean_wo+Y_std_wo, Y_mean_wo-Y_std_wo,
                     facecolor="#fda968", alpha=0.75)

    plt.plot(num_steps, Y_mean_w, lw=2,
             label="predicted (balanced)",
             color="#1b9e77", ls="-", mew=5)
    plt.fill_between(num_steps, Y_mean_w+Y_std_w, Y_mean_w-Y_std_w,
                     facecolor="#7ce9c8", alpha=0.75)

    plt.xlabel("time (s)", fontsize=16)
    plt.ylabel("steering angle (degree) ", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid()
    plt.show()

    abs_error = np.abs(Y_test)
    mean_error_wo = np.abs(Y_test-Y_mean_wo)
    mean_error_w = np.abs(Y_test-Y_mean_w)
    degree_idx = np.linspace(Y_test.min(), Y_test.max(), 50)

    error_wo_list = []
    error_w_list = []
    error_abs_list = []
    for idx in xrange(1, degree_idx.shape[0]-1):
        idx_up = (Y_test > degree_idx[idx-1])
        idx_down = (Y_test < degree_idx[idx+1])
        idices = np.logical_and(idx_up, idx_down)

        error_wo = np.mean(mean_error_wo[idices])
        error_w = np.mean(mean_error_w[idices])
        error_abs = np.mean(abs_error[idices])

        error_wo_list.append(error_wo)
        error_w_list.append(error_w)
        error_abs_list.append(error_abs)
    error_wo = np.array(error_wo_list)
    error_w = np.array(error_w_list)
    error_abs = np.array(error_abs_list)

    plt.figure()
    plt.plot(degree_idx[1:-1], error_wo/error_abs, lw=2,
             label="unbalanced",
             color="#d95f02", ls="-", mew=5)
    plt.plot(degree_idx[1:-1], error_w/error_abs, lw=2,
             label="balanced",
             color="#1b9e77", ls="-", mew=5, alpha=0.75)
    plt.xlabel("steering angle", fontsize=16)
    plt.ylabel("relative error (%)", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid()
    plt.show()


elif option == "get-loss-curve":
    # collect curves
    foryer_wo = collect_curves("foryer", "wo_balance", 10)
    #  foryer_w = collect_curves("foryer", "w_balance", 10)
    jogging_wo = collect_curves("jogging", "wo_balance", 10)
    jogging_w = collect_curves("jogging", "w_balance", 10)

    # collect test curves
    foryer_wo_mean, foryer_wo_std = get_test_loss_curve(foryer_wo)
    #  foryer_w_mean, foryer_w_std = get_test_loss_curve(foryer_w)
    jogging_wo_mean, jogging_wo_std = get_test_loss_curve(jogging_wo)
    jogging_w_mean, jogging_w_std = get_test_loss_curve(jogging_w)

    # plot curves
    num_steps = foryer_wo[0].shape[0]
    plt.figure()
    plt.plot(range(num_steps), foryer_wo_mean, lw=3,
             label="foyer unbalanced",
             color="#5C88DAFF", ls="-", mew=5)
    plt.fill_between(
        range(num_steps), foryer_wo_mean+foryer_wo_std,
        foryer_wo_mean-foryer_wo_std, facecolor="#5C88DA99")

    # jogging without balance
    plt.plot(range(num_steps), jogging_wo_mean, lw=3,
             label="jogging unbalanced",
             color="#CC0C00FF", ls="-", mew=5)
    plt.fill_between(
        range(num_steps), jogging_wo_mean+jogging_wo_std,
        jogging_wo_mean-jogging_wo_std, facecolor="#CC0C0099")

    # jogging balance
    plt.plot(range(num_steps), jogging_w_mean, lw=3,
             label="jogging balanced",
             color="#84BD00FF", ls="-", mew=5)
    plt.fill_between(
        range(num_steps), jogging_w_mean+jogging_w_std,
        jogging_w_mean-jogging_w_std, facecolor="#84BD0099")

    plt.xlabel("epoch", fontsize=16)
    plt.ylabel("loss", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.show()
