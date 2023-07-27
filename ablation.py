import warnings
warnings.filterwarnings('ignore')

import sys
import math
import numpy as np
import copy
import time
import pickle
from random import seed, shuffle, sample

from sklearn.preprocessing import StandardScaler
from aif360.datasets import AdultDataset, CompasDataset, GermanDataset, BankDataset, StandardDataset

import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from iFlipper.utils import measure_error, plot_result
from iFlipper import *

from configs import parse_ablation_comparison

from preprocess import get_dataset, get_sim_matrix
from iFlipper import ablation_comparison_methods

def main():
    opt = parse_ablation_comparison()
    if opt.verbose:
        print(opt)
        print()

    similarity_params, dset, plots = get_dataset(opt)
    x_train, y_train = dset.x_train, dset.y_train
    x_val, y_val = dset.x_val, dset.y_val
    x_test, y_test = dset.x_test, dset.y_test
    m_list = plots['m_list'][1:]

    # Obtain similarity matrix from dataset
    w_train, edge_train, w_edge_train = get_sim_matrix(opt, x_train, similarity_params)
    w_test, edge_test, w_edge_test = get_sim_matrix(opt, x_test, similarity_params)
    w_val, edge_val, w_edge_val = get_sim_matrix(opt, x_val, similarity_params)

    # Initial amount of total error
    total_error_arr, num_flips_arr, runtime = dict(), dict(), dict()

    init_error = measure_error(y_train, edge_train, w_edge_train)
    print(f"Initial number of violations: {init_error:.1f}")    

    # ["Greedy", "Gradient", "kMeans", "iFlipper", "ILP"]

    if "LP-SR" in opt.methods:
        method = "LP-SR"
        print(f"\n###### {method} begin")
        total_error_arr[method], num_flips_arr[method], runtime[method] = [], [], []

        for m in m_list:
            start = time.time()
            IFLIP = iFlipper(y_train, w_train, edge_train, w_edge_train, ablation_option = method)
            flipped_label = IFLIP.transform(m)
            elapsed_time = time.time() - start

            total_error = measure_error(flipped_label, edge_train, w_edge_train)
            num_flips = np.sum(y_train != flipped_label)

            total_error_arr[method].append(total_error)
            num_flips_arr[method].append(num_flips)
            runtime[method].append(elapsed_time)

            if opt.verbose:
                print("============================")
                print(f"Total error limit: {m:.1f}")
                print(f"Total error: {total_error:.1f}")
                print(f"Number of flips: {num_flips}")
                print(f"Runtime (sec): {elapsed_time:.5f}")

    # Greedy algorithm
    ## Flips labels that reduce the total error the most.
    if "LP-AR" in opt.methods:
        method = "LP-AR"
        print(f"\n###### {method} begin ######")
        total_error_arr[method], num_flips_arr[method], runtime[method] = [], [], []

        for m in m_list:
            start = time.time()
            IFLIP = iFlipper(y_train, w_train, edge_train, w_edge_train, ablation_option = method)
            flipped_label = IFLIP.transform(m)
            elapsed_time = time.time() - start

            total_error = measure_error(flipped_label, edge_train, w_edge_train)
            num_flips = np.sum(y_train != flipped_label)

            total_error_arr[method].append(total_error)
            num_flips_arr[method].append(num_flips)
            runtime[method].append(elapsed_time)

            if opt.verbose:
                print("============================")
                print(f"Total error limit: {m:.1f}")
                print(f"Total error: {total_error:.1f}")
                print(f"Number of flips: {num_flips}")
                print(f"Runtime (sec): {elapsed_time:.5f}")

    if "iFlipper" in opt.methods:
        method = "iFlipper"
        print(f"\n###### {method} begin")
        total_error_arr[method], num_flips_arr[method], runtime[method] = [], [], []
        for m in m_list:
            start = time.time()
            IFLIP = iFlipper(y_train, w_train, edge_train, w_edge_train, ablation_option = method)
            flipped_label = IFLIP.transform(m)
            elapsed_time = time.time() - start

            total_error = measure_error(flipped_label, edge_train, w_edge_train)
            num_flips = np.sum(y_train != flipped_label)

            total_error_arr[method].append(total_error)
            num_flips_arr[method].append(num_flips)
            runtime[method].append(elapsed_time)

            if opt.verbose:
                print("============================")
                print(f"Total error limit: {m:.1f}")
                print(f"Total error: {total_error:.1f}")
                print(f"Number of flips: {num_flips}")
                print(f"Runtime (sec): {elapsed_time:.5f}")

    # ILP Solver
    ## Solves the ILP problem exactly using CPLEX, which is a state-of-the-art solver.
    if "ILP" in opt.methods:
        method = "ILP"
        print(f"\n###### {method} begin ######")
        total_error_arr[method], num_flips_arr[method], runtime[method] = [], [], []

        for m in m_list:
            start = time.time()
            flipped_label = CPLEX_Solver(y_train, m, w_train, edge_train, ILP = True)
            elapsed_time = time.time() - start

            total_error = measure_error(flipped_label, edge_train, w_edge_train)
            num_flips = np.sum(y_train != flipped_label)

            total_error_arr[method].append(total_error)
            num_flips_arr[method].append(num_flips)
            runtime[method].append(elapsed_time)

            if opt.verbose:
                print("============================")
                print(f"Total error limit: {m:.1f}")
                print(f"Total error: {total_error:.1f}")
                print(f"Number of flips: {num_flips}")
                print(f"Runtime (sec): {elapsed_time:.5f}")

    plot_result(opt, m_list, total_error_arr, "Total Error", 1, target = "ablation", num_figures = 1)
    plot_result(opt, m_list, num_flips_arr, "# Flips", 0, target = "ablation", num_figures = 2)
    # plot_result(opt, m_list, runtime, "Runtime (sec)", 0, num_figures = 3)

# def plot_result(opt, m_list, performance_arr, y_axis, num_digits, target = "_", num_figures = 1):
#     labels = np.array(m_list)[::-1]

#     plot = list()
#     for k in performance_arr:
#         plot.append(np.array(performance_arr[k])[::-1])

#     x = np.arange(len(labels)) # the label locations
#     width = 0.17  # the width of the bars

#     plt.figure(num_figures, figsize=(20,15))
#     ax = plt.subplot()
#     [x.set_linewidth(2) for x in ax.spines.values()]

#     rts = list()
#     color_list = ["pink", "lightblue", "thistle", "darkgray", "wheat"]
#     hatch_list = ['//', '\\', '/', 'X', '.']
#     for i, e in enumerate(performance_arr):
#         rts.append(ax.barh(x + width * (2 - i), np.round(plot[i], num_digits), width, label=e, color=color_list[i], hatch=hatch_list[i],edgecolor="black", linewidth=2))

#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     plt.tick_params(labelsize=40)
#     ax.set_xlabel(y_axis, fontsize=45)
#     ax.set_ylabel("Total Error Limit (m)", fontsize=45)

#     ax.set_yticks(x)
#     ax.set_yticklabels(labels)
#     ax.set_xscale('log')
#     plt.legend(prop={'size':30}, bbox_to_anchor=(-0.025, 1), loc="lower left", ncol=5)
    
#     max_val, max_ratio = 0, 1.03
#     left, right = plt.xlim()
#     xmax = 10**(np.ceil(np.log10(right)*1.05))
#     xmax = 10**(np.log10(right)*1.10)
#     plt.xlim([1.0, xmax])
        
#     def autolabel(rects):
#         """Attach a text label above each bar in *rects*, displaying its height."""
#         for rect in rects:
#             ax.annotate('{}'.format(rect.get_width()),
#                         xy=(max(rect.get_width(), max_val)*max_ratio, rect.get_y() + rect.get_height()*0.285),
#                         fontsize=35)

#     for rt in rts:
#         autolabel(rt)

#     plt.tight_layout()
#     # plt.show()
#     name = y_axis.replace(" ","")
#     plt.savefig(f"{opt.save_directory}/ablation_comparison_{name}_{opt.dataset}_{opt.similarity_matrix}_{opt.model_type}.png")
#     print(f"{name}_{plt.axis()}")

if __name__ == '__main__':
    main()