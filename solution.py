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

from configs import parse_solution_comparison

from preprocess import get_dataset, get_sim_matrix

def main():
    opt = parse_solution_comparison()
    if opt.verbose:
        print(opt)
        print()

    similarity_params, dset, plots = get_dataset(opt)
    x_train, y_train = dset.x_train, dset.y_train
    x_val, y_val = dset.x_val, dset.y_val
    x_test, y_test = dset.x_test, dset.y_test
    m_list = plots['m_list']

    # Obtain similarity matrix from dataset
    w_train, edge_train, w_edge_train = get_sim_matrix(opt, x_train, similarity_params)
    w_test, edge_test, w_edge_test = get_sim_matrix(opt, x_test, similarity_params)
    w_val, edge_val, w_edge_val = get_sim_matrix(opt, x_val, similarity_params)

    # Initial amount of total error
    total_error_arr, num_flips_arr, runtime = dict(), dict(), dict()

    init_error = measure_error(y_train, edge_train, w_edge_train)
    print(f"Initial number of violations: {init_error:.1f}")    

    # ["Greedy", "Gradient", "kMeans", "iFlipper", "ILP"]

    # Greedy algorithm
    ## Flips labels that reduce the total error the most.
    if "Greedy" in opt.methods:
        method = "Greedy"
        print(f"\n###### {method} begin ######")
        total_error_arr[method], num_flips_arr[method], runtime[method] = [], [], []

        for m in m_list:
            start = time.time()
            flipped_label = Greedy(y_train, m, w_train, edge_train, w_edge_train)
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


    # Gradient-based algorithm
    ## Solves an unconstrained optimization problem via gradient descent
    if "Gradient" in opt.methods:
        method = "Gradient"
        print(f"\n###### {method} begin ######")
        total_error_arr[method], num_flips_arr[method], runtime[method] = [], [], []

        for m in m_list:
            start = time.time()
            flipped_label = Gradient(y_train, m, edge_train, w_edge_train, lam_high=60)
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


    # kMeans-based algorithm
    ## Applies k-means clustering and, for each cluster, make its examples have the majority label.
    if "kMeans" in opt.methods:
        method = "kMeans"
        print(f"\n###### {method} begin ######")
        total_error_arr[method], num_flips_arr[method], runtime[method] = [], [], []

        for m in m_list:
            start = time.time()
            flipped_label = kMeans(x_train, y_train, m, edge_train, w_edge_train)
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
            IFLIP = iFlipper(y_train, w_train, edge_train, w_edge_train)
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

    plot_result(opt, m_list, total_error_arr, "Total Error", 1, target = "solution", num_figures = 1)
    plot_result(opt, m_list, num_flips_arr, "# Flips", 0, target = "solution", num_figures = 2)
    plot_result(opt, m_list, runtime, "Runtime (sec)", 0, target = "solution", num_figures = 3)

# def plot_result(opt, m_list, performance_arr, y_axis, num_digits, num_figures = 1):
#     labels = np.array(m_list)[::-1]

#     plot1 = np.array(performance_arr["Greedy"])[::-1]
#     plot2 = np.array(performance_arr["Gradient"])[::-1]
#     plot3 = np.array(performance_arr["kMeans"])[::-1]
#     plot4 = np.array(performance_arr["iFlipper"])[::-1]
#     plot5 = np.array(performance_arr["ILP"])[::-1]

#     x = np.arange(len(labels)) # the label locations
#     width = 0.17  # the width of the bars

#     plt.figure(num_figures, figsize=(20,15))
#     ax = plt.subplot()
#     [x.set_linewidth(2) for x in ax.spines.values()]

#     rects1 = ax.barh(x + width * 2, np.round(plot1, num_digits), width, label="Greedy", color="pink", hatch='//',edgecolor="black", linewidth=2)
#     rects2 = ax.barh(x + width * 1, np.round(plot2, num_digits), width, label="Gradient", color="lightblue", hatch='\\',edgecolor="black", linewidth=2)
#     rects3 = ax.barh(x - width * 0, np.round(plot3, num_digits), width, label="kMeans", color="thistle", hatch='/',edgecolor="black", linewidth=2)
#     rects4 = ax.barh(x - width * 1, np.round(plot4, num_digits), width, label="iFlipper", color="darkgray", hatch='X',edgecolor="black", linewidth=2)
#     rects5 = ax.barh(x - width * 2, np.round(plot5, num_digits), width, label="ILP", color="wheat", edgecolor="black", linewidth=2)

#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     plt.tick_params(labelsize=40)
#     ax.set_xlabel(y_axis, fontsize=45)
#     ax.set_ylabel("Total Error Limit (m)", fontsize=45)

#     ax.set_yticks(x)
#     ax.set_yticklabels(labels)
#     ax.set_xscale('log')
#     plt.legend(prop={'size':30}, bbox_to_anchor=(-0.025, 1), loc="lower left", ncol=5)
    
#     max_val, max_ratio = 0, 1
#     if y_axis =="Total Error":
#         max_val = 118
#         plt.xlim(0, 40000)
#         max_ratio = 1.04
#     elif y_axis =="# Flips":
#         plt.minorticks_off()
#         plt.xlim(0, 3100)
#         max_ratio = 1.015
#     else:
#         plt.xlim(0, 6000)
#         max_ratio =1.03
        
#     def autolabel(rects):
#         """Attach a text label above each bar in *rects*, displaying its height."""
#         for rect in rects:
#             ax.annotate('{}'.format(rect.get_width()),
#                         xy=(max(rect.get_width(), max_val)*max_ratio, rect.get_y() + rect.get_height()*0.285),
#                         fontsize=35)
#     autolabel(rects1)
#     autolabel(rects2)
#     autolabel(rects3)
#     autolabel(rects4)
#     autolabel(rects5)

#     plt.tight_layout()
#     # plt.show()
#     name = y_axis.replace(" ","")
#     plt.savefig(f"{opt.save_directory}/solution_comparison_{name}_{opt.dataset}_{opt.similarity_matrix}_{opt.model_type}_.png")

if __name__ == '__main__':
    main()