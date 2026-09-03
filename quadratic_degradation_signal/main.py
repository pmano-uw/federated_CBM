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
    parser = argparse.ArgumentParser(description='Simulation params')

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--sim-num', type=int, default=30)
    parser.add_argument('--lap-noise', type=float, default=0)
    parser.add_argument('--savelog', action='store_true')
    parser.add_argument('--experiment', type=str, default='collaborative')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--window', type=int, default=1)

    input_args = parser.parse_args()

    args = {
        # Hierarchical graph params
        'delta': 1, 
        'mu_mu': np.array([[3], [1]]), 'Sigma_mu': np.array([[3, 0], [0, 1]]), # Prior of mu
        'shape_Sigma': np.array([2]) , 'scale_Sigma': np.array([0.5]), 'nu_Sigma': np.array([[5, 0], [0, 1]]), # Prior of Sigma
        'mu_Sigma': 0, 'Sigma_Sigma': 1, 'sigma0': np.array([2]), 'l_0': np.array([0]), 
        'mu_true': np.array([[1], [0.05]]), 'Sigma_true': np.array([[1e-4, 0], [0, 1e-6]]),
        'site_mu': np.ones(4).reshape(-1, 1) * 5, 'site_Sigma': np.eye(4) * 5, 'num_samples': 5000, 'num_burnin': 2500, 'num_chains': 15,
        # Simulation params
        'N': 50, 'M': 5, 'K': 80, 'S': 20, 's1_limit': 21, 'r_limit': 12, 'd': 2, 'max_T': 2000,
        # Value iteration params
        'c1':  -50, 'c2': -5, 'c3': -0.05, 'max_iter': 5000, 'threshold': 1e-3, 'gamma': 0.99, 'epsilon':1e-3,
        # Sampling parameters
        'gibbs_T': 2000,  'warm_up': 1000, 'n_chains': 4, 'space': 1
    }

    for k in vars(input_args):
       args[k] = vars(input_args)[k]

    print(args)

    # Define variables
    seed = args['seed']; random.seed(args['seed']); np.random.seed(args['seed'])

    # Run simulation
    if input_args.parallel:
        sim_list = tqdm(list(range(args['sim_num'])))
        results = Parallel(n_jobs=-1)(delayed(simulation)(args, i, args['experiment']) for i in sim_list)
    else:
        simulation(args, 0, args['experiment'])

    if args['savelog']:
        output_dir = "simulation_results/"
        details = f"{args['N']}_{args['M']}_{args['r_limit']}_{args['lap_noise']}_{args['experiment']}_"
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