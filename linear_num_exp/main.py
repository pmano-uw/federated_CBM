import numpy as np
# import pystan
import scipy
import argparse
import json

import time
import copy
import itertools

import matplotlib.pyplot as plt
import os
from simulation import run_sim_linear

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='store', type=str, default='nasa1', help="Select dataset between nasa or battery")
    parser.add_argument('--scenario', action='store', type=int, default=1, help="Select scenario")
    parser.add_argument('--num-site', type=int, default=60)

    input_args = parser.parse_args()
    args_linear = {
        # Hierarchical params
        'delta': 1, 'd': 1,
        'mu_mu': np.array([-10]), 'Sigma_mu': np.array([10]), # Prior of mu
        'alpha_Sigma': np.array([3]) , 'beta_Sigma': np.array([20]), # Prior of Sigma
        'sigma0': np.array([2]),
        'site_mu': np.array([[-10], [1]]), 'site_Sigma': np.array([[10, 0], [0, 1]]),
        # Simulation params
        'seed': 1234, 'sim_num': 30, 'epsilon': 1e-3, 'C': 5, 'N': 10, 'M': 5, 'r_limit': 0.7, 't_limit': 20,
        'prop_mu_mu': 10, 'prop_std_mu': 20, 'prop_mu_sigma': 0, 'prop_std_sigma': 1,
        'num_prop': 100000, 'num_target': 10000, 'ep_tol': 1e-3,
        # Sampling parameters
        'gibbs_T': 2000,  'warm_up': 1000, 'n_chains': 4, 'space': 1, "num_samples": 2500, "num_warmups": 1000
    }
    for k in vars(input_args):
       args_linear[k] = vars(input_args)[k]
    print(args_linear)
    # Set seed
    np.random.seed(args_linear['seed'])
    results = run_sim_linear(args_linear)

    output_dir = f"experiment_results/{args_linear['dataset']}_{args_linear['num_site']}_"
    jour = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    output_dir += jour
    result_dir = output_dir + ".json"
    with open(result_dir, 'w') as f:
        json.dump(results, f)