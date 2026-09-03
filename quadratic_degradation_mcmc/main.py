import numpy as np
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

from simulation import run_sim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel', action='store_true')
    input_args = parser.parse_args()
    
    args = {
        # Hierarchical params
        'delta': 1, 
        'mu_mu': np.array([[3], [1]]), 'Sigma_mu': np.array([[3, 0], [0, 1]]), # Prior of mu
        'shape_Sigma': np.array([2]) , 'scale_Sigma': np.array([1]), 'nu_Sigma': np.array([[5, 0], [0, 1]]), # Prior of Sigma
        'mu_Sigma': 0, 'Sigma_Sigma': 1, 'sigma0': np.array([0.25]), 'l_0': np.array([0]), 
        'mu_true': np.array([[1], [0.05]]), 'Sigma_true': np.array([[0.001, 0], [0, 1e-4]]),
        'site_mu': np.ones(5).reshape(-1, 1) * 5, 'site_Sigma': np.eye(5) * 5, 'num_samples': 2000, 'num_burnin': 1000, 'num_chains': 8,
        # Simulation params
        'N': 50, 'r_limit': 10, 'C': 5, 'd': 2, 'T': 30, 'seed': 1234, 'sim_num': 30, 'epsilon': 1e-3, 'ep_tol': 1e-3, 'rho_threshold': 0.08, 'max_iter': 10, 'damping': 0.1,
        # Sampling parameters
        'gibbs_T': 2000,  'warm_up': 1000, 'n_chains': 4, 'space': 1
    }
    # Set seed
    np.random.seed(args['seed'])
    args['max_lag'] = int((args['num_samples'] - args['num_burnin'])/10)
    # print(args)
    if input_args.parallel:
        total_cores = mp.cpu_count()
        site_jobs = max(1, int(np.floor(np.sqrt(total_cores))))
        sim_jobs = max(1, total_cores // site_jobs)
        args['site_n_jobs'] = site_jobs

        sim_list = tqdm(list(range(args['sim_num'])))
        results = Parallel(n_jobs=sim_jobs)(delayed(run_sim)(args, i) for i in sim_list)
    else:
        # Single simulation: no sim-level parallelism to compete with, so give the
        # site loop all the cores.
        args['site_n_jobs'] = -1
        results = [run_sim(args, 0)]

    output_dir = f"history/quad_{args['N']}_{args['sigma0'].item()}_{args['sim_num']}_"
    jour = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    output_dir += jour
    result_dir = output_dir + ".json"
    with open(result_dir, 'w') as f:
        json.dump(results, f)