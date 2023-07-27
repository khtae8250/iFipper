import argparse

import sys
import subprocess

from baselines import baseline_comparison_methods, runtime_comparison_methods
from iFlipper import ablation_comparison_methods

def parse_baseline(description = "argument for comparison"):
    parser = argparse.ArgumentParser(description)
    parser.add_argument('--dataset', type=str, default='AdultCensus', choices=['COMPAS', 'AdultCensus', "Credit"])
    parser.add_argument('--similarity_matrix', type=str, default='knn', choices=['knn', 'threshold'])
    parser.add_argument('--model_type', type=str, default='LogisticRegression', choices=['LogisticRegression', 'RandomForest', 'NeuralNetwork'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--target_baselines', type=str, default='all', help="write targeting baseline methods for comparison, default:all")
    parser.add_argument('--save_directory', type=str, default='./results', help="directory to save overall results")
    parser.add_argument('--verbose', action="store_true", help="print intermediate result")

    if parser.parse_known_args()[0].dataset == "COMPAS":
        parser.add_argument('--num_train', type=float, default=0.6)
        parser.add_argument('--num_test', type=float, default=0.3)
        parser.add_argument('--protected', type=str, default='sex', help="sensitive attribute for COMPAS, default:sex")
    elif parser.parse_known_args()[0].dataset == "AdultCensus":
        parser.add_argument('--num_train', type=float, default=0.6)
        parser.add_argument('--num_test', type=float, default=0.3)
        parser.add_argument('--protected', type=str, default='sex', help="sensitive attribute for AdultCensus, default:sex")
    elif parser.parse_known_args()[0].dataset == "Credit":
        parser.add_argument('--num_train', type=float, default=0.7)
        parser.add_argument('--num_test', type=float, default=0.2)
        parser.add_argument('--protected', type=str, default='age', help="sensitive attribute for Germen Credit, default:age")

    return parser


def parse_baseline_comparison():
    parser = parse_baseline("argument for baseline comparison")
    parser.add_argument('--load_LFR', type=bool, default=True, help="load pre-trained LFR model")
    parser.add_argument('--load_iFair', type=bool, default=True, help="load pre-trained iFair model")
    parser.add_argument('--path_LFR', type=str, default='default', help="path to pre-trained LFR model")
    parser.add_argument('--path_iFair', type=str, default='default', help="path to pre-trained LFR model")
    opt = parser.parse_args()

    if opt.target_baselines == "all":
        opt.methods = baseline_comparison_methods
    else:
        opt.methods = list()
        for method in baseline_comparison_methods:
            if method in opt.target_baselines:
                opt.methods.append(method)
        if len(opt.methods) == 0:
            print(f"target_baselines should contain these baselines: {baseline_comparison_methods}", file=sys.stderr)
            exit(1)

    # if opt.dataset == "COMPAS":
    #     opt.num_train = 0.6
    #     opt.num_test = 0.3
    #     opt.protected = "sex"

    # elif opt.dataset == "AdultCensus":
    #     opt.num_train = 0.6
    #     opt.num_test = 0.3
    #     opt.protected = "sex"

    # elif opt.dataset == "Credit":
    #     opt.num_train = 0.7
    #     opt.num_test = 0.2
    #     opt.protected = "age"
    
    if opt.path_LFR == "default":
        opt.path_LFR = f'./baselines/LFR/LFR_{opt.dataset}_{opt.num_train}.pkl'
    
    if opt.path_iFair == "default":
        opt.path_iFair = f'./baselines/iFair/iFair_{opt.dataset}_{opt.num_train}.pkl'

    return opt

def parse_solution_comparison():
    parser = parse_baseline("argument for runtime comparison")
    opt = parser.parse_args()

    if opt.target_baselines == "all":
        opt.methods = runtime_comparison_methods
    else:
        opt.methods = list()
        for method in runtime_comparison_methods:
            if method in opt.target_baselines:
                opt.methods.append(method)
        if len(opt.methods) == 0:
            print(f"target_baselines should contain these baselines: {runtime_comparison_methods}", file=sys.stderr)
            exit(1)

    return opt


def parse_ablation_comparison():
    parser = parse_baseline("argument for ablation comparison")
    opt = parser.parse_args()

    if opt.target_baselines == "all":
        opt.methods = ablation_comparison_methods
    else:
        opt.methods = list()
        for method in ablation_comparison_methods:
            if method in opt.target_baselines:
                opt.methods.append(method)
        if len(opt.methods) == 0:
            print(f"target_baselines should contain these baselines: {runtime_comparison_methods}", file=sys.stderr)
            exit(1)

    return opt


def parse_runtime_comparison():
    parser = argparse.ArgumentParser("argument for runtime comparison")
    parser.add_argument('--dataset', type=str, default='Synthetic', choices=["Synthetic"])
    parser.add_argument('--similarity_matrix', type=str, default='knn', choices=['knn', 'threshold'])
    parser.add_argument('--model_type', type=str, default='LogisticRegression', choices=['LogisticRegression', 'RandomForest', 'NeuralNetwork'])
    parser.add_argument('--seed', type=int, default=1122334455)
    parser.add_argument('--save_directory', type=str, default='./results', help="directory to save overall result")
    parser.add_argument('--verbose', action="store_true", help="print intermediate result")
    if parser.parse_known_args()[0].dataset == "Synthetic":
        parser.add_argument('--num_train', type=float, default=0.5)
        parser.add_argument('--num_test', type=float, default=0.3)
        # parser.add_argument('--protected', type=str, default='sex', help="sensitive attribute for COMPAS")


    opt = parser.parse_args()

    return opt