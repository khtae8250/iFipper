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

import warnings
# warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')
warnings.filterwarnings("ignore")

from aif360.algorithms.preprocessing import LFR
# from baselines.iFair.iFair import iFair
sys.path.append('./baselines/iFair/')
from iFair import iFair

from baselines.PFR.PFR import PFR, similarity_pfr, estimate_dim

# from iFlipper.iflipper import iFlipper
# from iFlipper.cplex_solver import CPLEX_Solver
# from iFlipper.greedy import Greedy
# from iFlipper.gradient import Gradient
# from iFlipper.kmeans import kMeans

# from iFlipper.utils import measure_error, generate_sim_matrix
# from iFlipper.model import Model

from iFlipper import *

from configs import parse_baseline_comparison

from preprocess import get_dataset, get_sim_matrix

def main():
    opt = parse_baseline_comparison()
    if opt.verbose:
        print(opt)
        print()
    model_type = opt.model_type

    similarity_params, dset, plots = get_dataset(opt)
    x_train, y_train = dset.x_train, dset.y_train
    x_val, y_val = dset.x_val, dset.y_val
    x_test, y_test = dset.x_test, dset.y_test
    num_plot = plots['num_plot']

    # Obtain similarity matrix from dataset
    w_train, edge_train, w_edge_train = get_sim_matrix(opt, x_train, similarity_params)
    w_test, edge_test, w_edge_test = get_sim_matrix(opt, x_test, similarity_params)
    w_val, edge_val, w_edge_val = get_sim_matrix(opt, x_val, similarity_params)

    # Gather results
    test_accuracy, test_consistency_score, runtime = dict(), dict(), dict()
    # baseline_comparison_methods = ["LFR", "iFair", "PFR", "iFlipper", "Original"]
    
    if "Original" in opt.methods:
        method = "Original"
        print(f"\n###### {method} begin ######")
        model = Model(model_type, x_train, y_train, edge_train, w_edge_train, x_test, y_test, edge_test, w_edge_test, x_val, y_val)
        train_performance, test_performance = model.train()

        test_accuracy[method], test_consistency_score[method] = [test_performance[0]], [test_performance[1]]
        print(f"{method} Test Accuracy: {test_performance[0]:.5f}")
        print(f"{method} Test Consistency Score: {test_performance[1]:.5f}")

    if "iFlipper" in opt.methods:
        method = "iFlipper"
        print(f"\n###### {method} begin")
        test_accuracy[method], test_consistency_score[method], runtime[method] = [], [], []
        init_error = measure_error(y_train, edge_train, w_edge_train)
        if opt.verbose:
            print(f"Initial amount of total error: {init_error:.1f}")

        for i in num_plot:
            m = (init_error * i)
                
            start = time.time()
            IFLIP = iFlipper(y_train, w_train, edge_train, w_edge_train)
            flipped_label = IFLIP.transform(m)
            elapsed_time = time.time() - start

            model = Model(model_type, x_train, flipped_label, edge_train, w_edge_train, x_test, y_test, edge_test, w_edge_test, x_val, y_val)
            train_performance, test_performance = model.train()

            test_accuracy[method].append(test_performance[0])
            test_consistency_score[method].append(test_performance[1])
            runtime[method].append(elapsed_time)
            
            if opt.verbose:
                print("============================")
                print(f"Total error limit: {m:.1f}")
                print(f"Total error: {measure_error(flipped_label, edge_train, w_edge_train):.1f}")
                print(f"Number of flips: {np.sum(y_train != flipped_label)}")
                print(f"Test Accuracy: {test_performance[0]:.5f}") 
                print(f"Test Consistency Score: {test_performance[1]:.5f}")
                print(f"Runtime (sec): {elapsed_time:.5f}")
    
    if "iFlipper" in opt.methods:
        method = "iFlipper"
        print(f"\n###### {method} begin ######")
        test_accuracy[method], test_consistency_score[method], runtime[method] = [], [], []

        init_error = measure_error(y_train, edge_train, w_edge_train)
        print(f"Initial amount of total error: {init_error:.1f}")

        for i in num_plot:
            m = (init_error * i)
                
            start = time.time()
            IFLIP = iFlipper(y_train, w_train, edge_train, w_edge_train)
            flipped_label = IFLIP.transform(m)
            elapsed_time = time.time() - start

            model = Model(model_type, x_train, flipped_label, edge_train, w_edge_train, x_test, y_test, edge_test, w_edge_test, x_val, y_val)
            train_performance, test_performance = model.train()

            test_accuracy[method].append(test_performance[0])
            test_consistency_score[method].append(test_performance[1])
            runtime[method].append(elapsed_time)
            
            if opt.verbose:
                print("============================")
                print(f"Total error limit: {m:.1f}")
                print(f"Total error: {measure_error(flipped_label, edge_train, w_edge_train):.1f}")
                print(f"Number of flips: {np.sum(y_train != flipped_label)}")
                print(f"Test Accuracy: {test_performance[0]:.5f}") 
                print(f"Test Consistency Score: {test_performance[1]:.5f}")
                print(f"Runtime (sec): {elapsed_time:.5f}")

    # LFR code is based on aif360 package
    if "LFR" in opt.methods:
        method = "LFR"
        print(f"\n###### {method} begin ######")
        test_accuracy[method], test_consistency_score[method], runtime[method] = [], [], []

        train = dset.train
        test = dset.test
        val = dset.val
        index = dset.index

        privileged_groups = [{opt.protected: 1}]
        unprivileged_groups = [{opt.protected: 0}]

        if opt.load_LFR:
            with open(opt.path_LFR, 'rb') as f:
                save_dict = pickle.load(f)
                
            grid = len(save_dict["Ax"])
            for i in range(grid):
                TR = save_dict["TR"][i]
                Ax = save_dict["Ax"][i]
                Ay = save_dict["Ay"][i]
                Az = save_dict["Az"][i]
                elapsed_time = save_dict["elapsed_time"][i]
                
                start = time.time()
                transf_train = TR.transform(train)
                transf_test = TR.transform(test)
                transf_val = TR.transform(val)

                transf_x_train = np.delete(transf_train.features, index, axis=1)
                transf_x_test = np.delete(transf_test.features, index, axis=1)
                transf_x_val = np.delete(transf_val.features, index, axis=1)
                elapsed_time = (time.time() - start) + elapsed_time

                model = Model(model_type, transf_x_train, y_train, edge_train, w_edge_train, transf_x_test, y_test, edge_test, w_edge_test, transf_x_val, y_val)
                train_performance, test_performance = model.train()

                test_accuracy[method].append(test_performance[0])
                test_consistency_score[method].append(test_performance[1])
                runtime[method].append(elapsed_time)

                if opt.verbose:
                    print("============================")
                    print(f"Ax:{Ax}, Ay:{Ay}, Az:{Az}")
                    print(f"Test Accuracy: {test_performance[0]:.5f}") 
                    print(f"Test Consistency Score: {test_performance[1]:.5f}")
                    print(f"Runtime (sec): {elapsed_time:.5f}")
        else:
            for Ax in [0.01]:
                for Ay in [0.1, 0.5, 1, 5]:
                    for Az in [0, 0.1, 0.5, 1, 5]:
                        start = time.time()
                        TR = LFR(unprivileged_groups=unprivileged_groups,
                                    privileged_groups=privileged_groups,
                                    Ax=Ax, Ay=Ay, Az=Az,
                                    verbose=0)
                        TR = TR.fit(train)

                        transf_train = TR.transform(train)
                        transf_test = TR.transform(test)
                        transf_val = TR.transform(val)

                        transf_x_train = np.delete(transf_train.features, index, axis=1)
                        transf_x_test = np.delete(transf_test.features, index, axis=1)
                        transf_x_val = np.delete(transf_val.features, index, axis=1)
                        elapsed_time = (time.time() - start)

                        model = Model(model_type, transf_x_train, y_train, edge_train, w_edge_train, transf_x_test, y_test, edge_test, w_edge_test, transf_x_val, y_val)
                        train_performance, test_performance = model.train()

                        test_accuracy[method].append(test_performance[0])
                        test_consistency_score[method].append(test_performance[1])
                        runtime[method].append(elapsed_time)

                        if opt.verbose:
                            print("============================")
                            print(f"Ax:{Ax}, Ay:{Ay}, Az:{Az}")
                            print(f"Test Accuracy: {test_performance[0]:.5f}") 
                            print(f"Test Consistency Score: {test_performance[1]:.5f}")
                            print(f"Runtime (sec): {elapsed_time:.5f}")

    if "iFair" in opt.methods:
        method = "iFair"
        print(f"\n###### {method} begin ######")
        test_accuracy[method], test_consistency_score[method], runtime[method] = [], [], []

        x_train_with_sensitive = dset.x_train_with_sensitive
        x_test_with_sensitive = dset.x_test_with_sensitive
        x_val_with_sensitive = dset.x_val_with_sensitive


        if opt.load_iFair:
            with open(opt.path_iFair, 'rb') as f:
                save_dict = pickle.load(f)

            grid = len(save_dict["Ax"])
            for i in range(grid):
                iF = save_dict["iF"][i]
                Ax = save_dict["Ax"][i]
                Az = save_dict["Az"][i]
                k = save_dict["k"][i]
                elapsed_time = save_dict["elapsed_time"][i]
                
                start = time.time()
                transf_x_train = iF.transform(x_train_with_sensitive)
                transf_x_test = iF.transform(x_test_with_sensitive)
                transf_x_val = iF.transform(x_val_with_sensitive)

                transf_x_train = np.delete(transf_x_train, -1, axis=1)
                transf_x_test = np.delete(transf_x_test, -1, axis=1)
                transf_x_val = np.delete(transf_x_val, -1, axis=1)
                elapsed_time = (time.time() - start) + elapsed_time

                model = Model(model_type, transf_x_train, y_train, edge_train, w_edge_train, transf_x_test, y_test, edge_test, w_edge_test, transf_x_val, y_val)
                train_performance, test_performance = model.train()

                test_accuracy[method].append(test_performance[0])
                test_consistency_score[method].append(test_performance[1])
                runtime[method].append(elapsed_time)

                if opt.verbose:
                    print("============================")
                    print(f"Ax:{Ax}, Az:{Az}, k:{k}")
                    print(f"Test Accuracy: {test_performance[0]:.5f}") 
                    print(f"Test Consistency Score: {test_performance[1]:.5f}")
                    print(f"Runtime (sec): {elapsed_time:.5f}")

        else:
            for Ax in [0.01]:
                for Az in [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 10, 20, 100, 200, 1000, 2000]:
                    for k in [10, 20, 30]:
                        start = time.time()
                        iF = iFair(k=k, A_x = Ax, A_z = Az, max_iter = 10000, nb_restarts=3)
                        iF.fit(x_train_with_sensitive)

                        transf_x_train = iF.transform(x_train_with_sensitive)
                        transf_x_test = iF.transform(x_test_with_sensitive)
                        transf_x_val = iF.transform(x_val_with_sensitive)

                        transf_x_train = np.delete(transf_x_train, -1, axis=1)
                        transf_x_test = np.delete(transf_x_test, -1, axis=1)
                        transf_x_val = np.delete(transf_x_val, -1, axis=1)
                        elapsed_time = time.time() - start

                        model = Model(model_type, transf_x_train, y_train, edge_train, w_edge_train, transf_x_test, y_test, edge_test, w_edge_test, transf_x_val, y_val)
                        train_performance, test_performance = model.train()

                        test_accuracy[method].append(test_performance[0])
                        test_consistency_score[method].append(test_performance[1])
                        runtime[method].append(elapsed_time)

                        if opt.verbose:
                            print("============================")
                            print(f"Ax:{Ax}, Az:{Az}, k:{k}")
                            print(f"Test Accuracy: {test_performance[0]:.5f}") 
                            print(f"Test Consistency Score: {test_performance[1]:.5f}")
                            print(f"Runtime (sec): {elapsed_time:.5f}")

    if "PFR" in opt.methods:
        method = "PFR"
        print(f"\n###### {method} begin ######")
        test_accuracy[method], test_consistency_score[method], runtime[method] = [], [], []

        k_dim = estimate_dim(x_train)
        w_pfr = similarity_pfr(x_train, k)

        for gamma in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            start = time.time()
            PFR_model = PFR(k = k_dim, W_s = w_pfr, W_F = w_train, gamma=gamma)
            PFR_model.fit(x_train)

            transf_x_train = PFR_model.transform(x_train)
            transf_x_test = PFR_model.transform(x_test)
            transf_x_val = PFR_model.transform(x_val)
            elapsed_time = time.time() - start
            
            model = Model(model_type, transf_x_train, y_train, edge_train, w_edge_train, transf_x_test, y_test, edge_test, w_edge_test, transf_x_val, y_val)
            train_performance, test_performance = model.train()

            test_accuracy[method].append(test_performance[0])
            test_consistency_score[method].append(test_performance[1])
            runtime[method].append(elapsed_time)

            if opt.verbose:
                print("============================")
                print("Gamma: %s" % gamma)
                print(f"Test Accuracy: {test_performance[0]:.5f}") 
                print(f"Test Consistency Score: {test_performance[1]:.5f}")
                print(f"Runtime (sec): {elapsed_time:.5f}")

    # Draw trade-off curve and save
    from matplotlib.ticker import FormatStrFormatter

    methods = ["LFR", "iFair", "PFR", "iFlipper", "Original"]
    shapes = ["o", "^", "P", "D", "X"]
    marker_size = 700

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    for i in range(len(methods)):
        if methods[i] == "Original":
            plt.scatter(test_accuracy[methods[i]], test_consistency_score[methods[i]], label=methods[i], s=marker_size+100, marker=shapes[i], edgecolors="black", linewidth=2, color="white")
        else:
            plt.scatter(test_accuracy[methods[i]], test_consistency_score[methods[i]], label=methods[i], s=marker_size, marker=shapes[i]) 
            
    plt.tick_params(labelsize=35)
    plt.xlabel("Test Accuracy", fontsize=45)
    plt.ylabel("Test Consistency Score", fontsize=45)
    plt.legend(prop={'size':35})

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{opt.save_directory}/baseline_comparison_{opt.dataset}_{opt.similarity_matrix}_{opt.model_type}.png")

if __name__ == '__main__':
    main()