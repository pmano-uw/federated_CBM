import numpy as np
# import pystan
import scipy
import argparse
import json

import time
import copy
import itertools

import multiprocessing as mp
from tqdm import tqdm
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import os
from simulation import run_sim_linear

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--adv-node', type=int, default=0)
    parser.add_argument('--adv-noise', type=float, default=0.0)
    input_args = parser.parse_args()
    args_linear = {
        # Hierarchical params
        'delta': 1, 
        'mu_mu': np.array([10]), 'Sigma_mu': np.array([10]), # Prior of mu
        'alpha_Sigma': np.array([3]) , 'beta_Sigma': np.array([20]), # Prior of Sigma
        'sigma0': np.array([2]), 'l_0': np.array([0]), 
        'mu_true': np.array([2]), 'Sigma_true': np.array([0.001]),
        'site_mu': np.array([[10], [1]]), 'site_Sigma': np.array([[10, 0], [0, 1]]),
        # Simulation params
        'N': 50, 'r_limit': 10, 'C': 5, 'd': 1, 'T': 70, 'seed': 1234, 'sim_num': 30, 'epsilon': 1e-3,
        'prop_mu_mu': 10, 'prop_std_mu': 20, 'prop_mu_sigma': 0, 'prop_std_sigma': 1,
        'num_prop': 100000, 'num_target': 10000, 'ep_tol': 1e-3,
        # Sampling parameters
        'gibbs_T': 2000,  'warm_up': 1000, 'n_chains': 4, 'space': 1, "num_samples": 2500, "num_warmups": 1000
    }
    
    for k in vars(input_args):
       args_linear[k] = vars(input_args)[k]
    # Set seed
    np.random.seed(args_linear['seed'])
    
    if input_args.parallel:
        sim_list = tqdm(list(range(args_linear['sim_num'])))
        results = Parallel(n_jobs=-1)(delayed(run_sim_linear)(args_linear, i) for i in sim_list)
    else:
        run_sim_linear(args_linear, 0)

    # Format adv_noise for filenames: convert to string and replace '.' with '_'
    noise_str = str(args_linear['adv_noise']).replace('.', '_')
    output_dir = f"history/{args_linear['N']}_{noise_str}_{args_linear['adv_node']}_"
    jour = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    output_dir += jour
    result_dir = output_dir + ".json"
    with open(result_dir, 'w') as f:
        json.dump(results, f)