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

from iFlipper.utils import measure_error, generate_sim_matrix, generate_original_sim_matrix

from iFlipper import *

from configs import parse_runtime_comparison

from preprocess import get_dataset, generate_synthetic_data

def main():
    opt = parse_runtime_comparison()
    if opt.verbose:
        print(opt)
        print()

    similarity_params, dset, plots = get_dataset(opt)
    x_train, y_train = dset.x_train, dset.y_train
    x_val, y_val = dset.x_val, dset.y_val
    x_test, y_test = dset.x_test, dset.y_test
    size_arr = plots['size_arr']

    # Initial amount of total error
    test_accuracy, test_consistency_score, runtime = dict(), dict(), dict()

    for method in size_arr:
        
        size = 2 * method
        num_examples1, num_examples2 = int(0.75 * size), int(0.25 * size)

        X, Y = generate_synthetic_data(opt, num_examples1, num_examples2)

        num_train, num_test, num_val = 0.5, 0.3, 0.2

        x_train, x_test, x_val = X[:int(num_train*len(Y))], X[int(num_train*len(Y)):int(num_train*len(Y))+int(num_test*len(Y))], X[int(num_train*len(Y))+int(num_test*len(Y)):]
        y_train, y_test, y_val = Y[:int(num_train*len(Y))], Y[int(num_train*len(Y)):int(num_train*len(Y))+int(num_test*len(Y))], Y[int(num_train*len(Y))+int(num_test*len(Y)):]

        w_train, edge_train, w_edge_train = generate_original_sim_matrix(x_train, opt.similarity_matrix, similarity_params)
        w_test, edge_test, w_edge_test = generate_original_sim_matrix(x_test, opt.similarity_matrix, similarity_params)
        w_val, edge_val, w_edge_val = generate_original_sim_matrix(x_val, opt.similarity_matrix, similarity_params)

        print(x_train.shape, x_test.shape, x_val.shape)
        print(len(w_train[w_train > 0]), len(edge_train), measure_error(y_train, edge_train, w_edge_train))

        model = Model(opt.model_type, x_train, y_train, edge_train, w_edge_train, x_test, y_test, edge_test, w_edge_test, x_val, y_val)
        train_performance, test_performance = model.train()

        init_error = measure_error(y_train, edge_train, w_edge_train)

        if opt.verbose:
            print("============================")
            print(f"Test Accuracy: {test_performance[0]:.5f}") 
            print(f"Test Consistency Score: {test_performance[1]:.5f}")
            print(f"Initial amount of total error: {init_error:.1f}")

        for i in [0.2]:
            m = (init_error * i)
                
            start = time.time()
            IFLIP = iFlipper(y_train, w_train, edge_train, w_edge_train)
            flipped_label = IFLIP.transform(m)
            elapsed_time = time.time() - start

            model = Model(opt.model_type, x_train, flipped_label, edge_train, w_edge_train, x_test, y_test, edge_test, w_edge_test, x_val, y_val)
            train_performance, test_performance = model.train()

            if opt.verbose:
                print("============================")
                print(f"Total error limit: {m:.1f}")
                print(f"Total error: {measure_error(flipped_label, edge_train, w_edge_train):.1f}")
                print(f"Number of flips: {np.sum(y_train != flipped_label)}")
                print(f"Test Accuracy: {test_performance[0]:.5f}") 
                print(f"Test Consistency Score: {test_performance[1]:.5f}")
                print(f"Runtime (sec): {elapsed_time:.5f}")

            test_accuracy[method] = test_performance[0]
            test_consistency_score[method] = test_performance[1]
            runtime[method] = elapsed_time
            # print(np.sum(flipped_label == 1), np.sum(flipped_label == 0))
            # plot_data(x_train, flipped_label)

    runtime_arr = []
    for key in runtime:
        runtime_arr.append(runtime[key])

    plt.figure(figsize=(12, 8))
    plt.scatter(size_arr, runtime_arr, s=300, marker='D')
    plt.plot(size_arr, runtime_arr, linewidth=3)

    plt.tick_params(labelsize=30)
    plt.ylabel("Runtime (sec)", fontsize=35)
    plt.xlabel("# Training Data", fontsize=35)

    plt.tight_layout()
    # plt.savefig(f"synthetic.pdf")
    # plt.show()
    plt.savefig(f"{opt.save_directory}/runtime_comparison.png")


def plot_data(opt, X, y):
    plt.figure(figsize=(12, 8))
    num_to_draw = 200 # we will only draw a small number of points to avoid clutter
    plt.scatter(X[y==1.0][:, -2], X[y==1.0][:, -1], color='green', marker='o', s=50, linewidth=2)
    plt.scatter(X[y==0.0][:, -2], X[y==0.0][:, -1], color='red', marker='x', s=50, linewidth=2)
    plt.legend(loc=2, fontsize=15)
    # plt.show()
    plt.savefig(f"{opt.save_directory}/runtime_comparison_{opt.dataset}_{opt.similarity_matrix}_{opt.model_type}.png")

if __name__ == '__main__':
    main()