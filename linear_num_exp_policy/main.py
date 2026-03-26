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
    parser.add_argument('--ep-show-console', action='store_true')
    parser.add_argument('--ep-log-site', action='store_true')
    parser.add_argument('--output-dir', type=str, default='experiment_results', help='Base directory for saving logs')

    input_args = parser.parse_args()

    args = {
        # Hierarchical graph params
        'delta': 1, 'mu_mu': np.array([10]), 'Sigma_mu': np.array([10]), # Prior of muS
        'alpha_Sigma': np.array([3]) , 'beta_Sigma': np.array([20]), # Prior of Sigma
        'l_0': np.array([0]), 
        'mu_true': np.array([1]), 'Sigma_true': np.array([0.01]),
        'site_mu': np.array([[10], [1]]), 'site_Sigma': np.array([[10, 0], [0, 1]]),
        # Simulation params
        'N': 50, 'M': 2, 'K': 60, 'S': 50, 'C': 3, 's1_limit': 16, 'r_limit': 15, 'd': 1, 'max_T': 5000,
        'prop_mu_mu': 5, 'prop_std_mu': 5, 'prop_mu_sigma': 0, 'prop_std_sigma': 1,
        'num_prop': 100000, 'num_target': 75000, 'ep_tol': 1e-3, 'initial_t': 5,
        # Value iteration params
        'c1':  -50, 'c2': -5, 'c3': -0.05, 'max_iter': 5000, 'threshold': 1e-3, 'gamma': 0.99, 'epsilon':1e-3,
        'window': 1,
        # Sampling parameters
        'gibbs_T': 2000,  'warm_up': 1000, 'n_chains': 4, 'space': 1
    }

    for k in vars(input_args):
       args[k] = vars(input_args)[k]

    # print(args)

    # Run simulation
    num_repetitions = 30
    print(f"Starting {num_repetitions} parallel simulation runs for the {args['experiment']} experiment...")

    # n_jobs=-1 tells joblib to use all available CPU threads
    # delayed() wraps your function so it can be distributed across the cores
    all_results = Parallel(n_jobs=-1)(
        delayed(simulation)(args, sim_round, args['experiment']) for sim_round in range(num_repetitions)
    )

    end_time = time.time()
    print(f"Run time: {(end_time - start_time)/60:.4f} mins")

    if args['savelog']:
        output_dir = args['output_dir']
        details = f"{args['N']}_{args['M']}_{args['r_limit']}_{args['lap_noise']}_{args['dataset']}_{args['experiment']}_"
        # jour = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        output_dir = os.path.join(output_dir, details)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        result_dir = os.path.join(output_dir, "result.json")
        params_dir = os.path.join(output_dir, "argument.json")
        
        # Dump the entire list of 30 result dictionaries into the JSON
        with open(result_dir, 'w') as f:
            json.dump(all_results, f, indent=2, default=int)

        for k in args:
            if isinstance(args[k], np.ndarray):
                args[k] = args[k].tolist()

        with open(params_dir, 'w') as f:
            json.dump(args, f)