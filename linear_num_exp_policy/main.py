import argparse
import random
import numpy as np
import time
import os

import json

from tqdm import tqdm
from joblib import Parallel, delayed 

from simulation import simulation

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Simulation params')

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--savelog', action='store_true')
    parser.add_argument('--experiment', type=str, default='collaborative')
    parser.add_argument('--lap-noise', type=float, default=0)
    parser.add_argument('--dataset', action='store', type=str, default='nasa1', help="Select dataset between nasa dataset")
    parser.add_argument('--num-site', type=int, default=60)

    input_args = parser.parse_args()

    args = {
        # Hierarchical graph params
        'delta': 1, 'mu_mu': np.array([10]), 'Sigma_mu': np.array([10]), # Prior of mu
        'alpha_Sigma': np.array([3]) , 'beta_Sigma': np.array([20]), # Prior of Sigma
        'sigma0': np.array([4]), 'l_0': np.array([0]), 
        'mu_true': np.array([1]), 'Sigma_true': np.array([0.01]),
        'site_mu': np.array([[10], [1]]), 'site_Sigma': np.array([[10, 0], [0, 1]]),
        # Simulation params
        'N': 100, 'M': 1, 'K': 60, 'S': 50, 'C': 3, 's1_limit': 9, 'r_limit': 8, 'd': 1, 'max_T': 5000,
        'prop_mu_mu': 5, 'prop_std_mu': 5, 'prop_mu_sigma': 0, 'prop_std_sigma': 1,
        'num_prop': 100000, 'num_target': 75000, 'ep_tol': 1e-5, 'initial_t': 5,
        # Value iteration params
        'c1':  -50, 'c2': -5, 'c3': -0.05, 'max_iter': 5000, 'threshold': 1e-3, 'gamma': 0.99, 'epsilon':1e-3,
        'window': 1,
        # Sampling parameters
        'gibbs_T': 2000,  'warm_up': 1000, 'n_chains': 4, 'space': 1
    }

    for k in vars(input_args):
       args[k] = vars(input_args)[k]

    print(args)

    # Define variables
    seed = args['seed']; random.seed(args['seed']); np.random.seed(args['seed'])

    # Run simulation
    results = simulation(args, 0, args['experiment'])
    end_time = time.time()
    print(f"Run time: {(end_time - start_time)/60} mins")

    if args['savelog']:
        output_dir = "experiment_results/"
        details = f"{args['N']}_{args['M']}_{args['r_limit']}_{args['lap_noise']}_{args['dataset']}_{args['experiment']}_"
        jour = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        output_dir += details
        output_dir += jour
        # print(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        result_dir = output_dir + "/result.json"
        params_dir = output_dir + "/argument.json"
        with open(result_dir, 'w') as f:
            json.dump(results, f, indent=2, default=int)

        for k in args:
            if isinstance(args[k], np.ndarray):
                args[k] = args[k].tolist()

        with open(params_dir, 'w') as f:
            json.dump(args, f)