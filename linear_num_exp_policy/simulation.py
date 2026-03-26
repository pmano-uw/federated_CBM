import itertools
import time
import copy
import random
import scipy 
from scipy import stats

import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

from value_iteration import value_iteration, VI_opt, calc_transition_prob_opt
from posterior import reset_EP_priors, update_EP_posterior, update_isolated_posterior, update_posterior, device_posterior_update
from utilities import calc_cost, suppress_stdout_stderr, load_data, process_dataframe, extract_prefix_and_number

def discretize_states(lk_list, bins):
    discrete_state_space = np.digitize(lk_list, bins, right=False)
    return discrete_state_space

def simulation(args, sim_round, experiment):
    # Define variables
    run_seed = int(args['seed']) + int(sim_round)
    random.seed(run_seed)
    np.random.seed(run_seed)
    rng = np.random.default_rng(run_seed)

    print(f"Running simulation #{sim_round}")
    d = args['d']

    # Create bins to discretize states
    dataset_name, dataset_num = extract_prefix_and_number(args['dataset'])
    bins_state_1 = np.linspace(args['l_0'], args['s1_limit'], args['S']-1).flatten()

    # Import data for all the sites
    df = load_data(args)

    # Calculate estimated variance
    burn_in_data = df.iloc[:args['initial_t'], :]
    increments = burn_in_data.diff()
    
    all_increments = increments.values.flatten()
    all_increments = all_increments[~np.isnan(all_increments)]
    
    # Ensure we have enough data points to calculate variance, otherwise use a fallback
    if len(all_increments) > 1:
        empirical_variance = np.var(all_increments, ddof=1)
        sigma_0 = np.array([np.sqrt(empirical_variance)])
    else:
        # Fallback to an arbitrary small prior if initial_t is 1
        sigma_0 = np.array([1.0]) 

    args['sigma0'] = sigma_0
    args['sigma0_initial'] = sigma_0.item() # Keep this clean fallback copy

    # Specify the max length of transition
    N_data = df.shape[1] 

    hist_lk_tensor = np.zeros((args['N'], args['M'], args['K']))
    hist_diff_lk_tensor = np.zeros((args['N'], args['M'], args['K']))
    hist_cost_tensor = np.zeros((args['N'], args['M'], args['K']))
    discrete_state_1 = np.zeros((args['N'], args['M'], args['K']))

    # Build a heterogeneity-reduced split across sites:
    # rank units by degradation trend, then distribute round-robin so each site gets a mixed profile.
    if args['N'] * args['M'] > N_data:
        raise ValueError(f"N*M={args['N'] * args['M']} exceeds available units={N_data}")

    effective_K = min(args['K'], df.shape[0])
    unit_values = df.iloc[:effective_K, :].to_numpy()
    denom = max(effective_K - 1, 1)
    degradation_score = (unit_values[-1, :] - unit_values[0, :]) / denom
    sorted_idx = np.argsort(degradation_score)[:args['N'] * args['M']]

    idx_list = [[] for _ in range(args['N'])]
    for pos, unit_idx in enumerate(sorted_idx):
        site = pos % args['N']
        idx_list[site].append(int(unit_idx))
    # print(idx_list)
    # Process data
    for n in range(args['N']):
        for m in range(args['M']):
            # Add noise
            if experiment == 'collaborative' and args['lap_noise'] > 0:
                noise = rng.laplace(scale=args['lap_noise'], size=args['K'])
            else:
                noise = np.zeros(args['K'])
            hist_lk_tensor[n, m, :] = df.iloc[:args['K'], idx_list[n][m]]
            # print(hist_lk_tensor[n, m, :])
            hist_diff_lk_tensor[n, m, 1:] = np.diff(hist_lk_tensor[n, m, :]) 
            hist_diff_lk_tensor[n, m, 0] = hist_lk_tensor[n, m, 0]

            # Extract states
            state_1 = args['delta'] * hist_lk_tensor

            # Discretize states
            discrete_state_1[n, m, :] = discretize_states(state_1[n, m, :], bins_state_1)
            
    # print("Finish generating data")
    discrete_state_1 = discrete_state_1.astype(int)
    
    # Running posterior update
    m_counter = np.zeros(args['N']).astype(int)
    k_counter = np.ones(args['N']).astype(int)

    hist_cost_list = [[] for n in range(args['N'])]
    hist_value_gap = [[] for n in range(args['N'])]
    hist_policy_gap = [[] for n in range(args['N'])]
    hist_prob_gap = [[] for n in range(args['N'])]
    
    hist_lk_diff_list = [[] for n in range(args['N'])]
    hist_lk_list = [[] for n in range(args['N'])]

    pi_hist = []
    flag_hist = []
    mu_mean_hist = []
    mu_cov_hist = []
    value_hist = []
    
    values = np.zeros((args['N'], args['K'], args['S']))
    pis = np.zeros((args['N'], args['K'], args['S']))
    
    opt_values = np.zeros((args['N'], args['K'], args['S']))
    values_under_opt_trans = [[] for n in range(args['N'])]
    opt_pis = np.zeros((args['N'], args['K'], args['S']))
    
    # Set parameters for EP
    if args['experiment'] == 'EP':
        # Set expected values
        mu_Sigma = args['beta_Sigma'] / (args['alpha_Sigma'] - args['d'] - 1)
        ev_initial = np.concatenate((args['mu_mu'].flatten(), mu_Sigma.flatten())).reshape(-1, 1)

        # Set covariance
        psi = copy.deepcopy(args['beta_Sigma'])
        nu = copy.deepcopy(args['alpha_Sigma'])
        var_Sigma = args['beta_Sigma']**2 / ((nu-1)**2 * (nu-2))
        cov_initial = scipy.linalg.block_diag(args['Sigma_mu']**2, var_Sigma)

        # Specify the model
        with suppress_stdout_stderr():
            hybrid_dist_model = CmdStanModel(stan_file="stan_file/hybrid_posterior.stan", model_name=f"hybrid_distribution")
            predictive_model = CmdStanModel(stan_file="stan_file/predictive_posterior.stan", model_name=f"predictive_posterior")

        # Initialize EP approximation once and keep updating it across windows.
        r, Q, r_list, Q_list = reset_EP_priors(ev_initial, cov_initial, args)

    new_bins = copy.deepcopy(bins_state_1)
    new_bins[1:] = bins_state_1[1:] - np.diff(bins_state_1)[0] / 2
    new_bins = np.append(new_bins, args['s1_limit'])

    # Derive optimal policy for each site
    for t in range(args['max_T']):     
        updated = 0
        # Check termination condition
        if np.all(m_counter == args['M']):
            break
        for i in range(args['N']):  
            # Check update skip condition
            if m_counter[i] == args['M']:
                continue
                
            m = m_counter[i]; m_idx = m - 1; k = k_counter[i] # Variables
            c_state_1 = state_1[i, m_idx, k] # States
            d_state_1 = discrete_state_1[i, m_idx, k];  # Discrete states

            # Retrieve pi
            pi = pis[i, k, :]
            
            # Diff lk
            diff_lk = hist_diff_lk_tensor[i, m_idx, k]
            lk = hist_lk_tensor[i, m_idx, k]

            # Save history
            hist_lk_diff_list[i].append([k, diff_lk.item()])
            hist_lk_list[i].append([k, lk.item()])
            
            # Determine the action from pi (policy) and replacement limit
            if (c_state_1 > args['r_limit'] or pi[d_state_1] > 0.5 or k_counter[i] == args['K']-1) and t >= args['initial_t']:
                cost_type = 'urgent' if c_state_1 > args['r_limit'] else 'schedule'
                cost = calc_cost(k, args, cost_type)
               
                # Append the cost
                hist_cost_list[i].append([k, cost])
                
                # Increment the counters
                m_counter[i] += 1
                k_counter[i] = 0
                print(f"Round {sim_round} | t = {t} | k = {k} | site = {i} | Count: {m_counter[:10]} | cost = {cost:.6f} | {c_state_1:.4f} | {d_state_1}")
                # print(np.mean(np.abs(mu0_mean - mu1_true)))
                # print(pi)
                # print('-'*30)
                
        if t % args['window'] == 0:
            if experiment == 'collaborative': 
                mu0_mean, mu0_cov = update_posterior(args, hist_lk_diff_list, noise, rng)
            elif experiment == 'isolated':
                # Calculate site-specific sigma_0
                local_sigma0_arr = np.zeros(args['N'])
                for site in range(args['N']):
                    local_diffs = [item[1] for item in hist_lk_diff_list[site]]
                    # Require at least 2 points to calculate standard deviation
                    if len(local_diffs) > 1:
                        local_sigma0_arr[site] = np.std(local_diffs, ddof=1)
                    else:
                        local_sigma0_arr[site] = args['sigma0_initial']
                
                # Assign the array to args for the posterior update
                args['sigma0_array'] = local_sigma0_arr
                mu0_mean, mu0_cov = update_isolated_posterior(args, hist_lk_diff_list)

            elif experiment == 'EP':
                r, Q, r_list, Q_list = update_EP_posterior(r, Q, r_list, Q_list, hist_lk_diff_list, hybrid_dist_model, args, rng)
                mu0_mean, mu0_cov = device_posterior_update(r, Q, r_list, Q_list, hist_lk_diff_list, predictive_model, args, rng)

            # Update sample variance
            data_list = []

            mu_mean_hist.append(mu0_mean.tolist())
            mu_cov_hist.append(mu0_cov.tolist())
            
            # Run VI for each site
            for site in range(args['N']):
                # Inject local sigma_0 for this specific site's SMDP transition probabilities
                if experiment == 'isolated' and 'sigma0_array' in args:
                    args['sigma0'] = np.array([args['sigma0_array'][site]])

                values[site], pis[site], probs = value_iteration(mu0_mean[site], mu0_cov[site], bins_state_1, args)
                
                # aggregate globally if we are doing a collaborative/EP run
                if experiment != 'isolated':
                    for j in range(len(hist_lk_diff_list[site])):
                        data_list.append(hist_lk_diff_list[site][j][1])
            
            # Calc global sample variance ONLY for collaborative/EP
            if experiment != 'isolated' and len(data_list) > 1:
                args['sigma0'] = np.array([np.std(data_list, ddof=1)])

        # Increment k at the end of every sweep
        k_counter += 1 
        
    results = {
        'hist_cost': hist_cost_list,
        # 'hist_value_gap': hist_value_gap,
        # 'hist_policy_gap': hist_policy_gap,
        'mu_mean_hist': mu_mean_hist,
        'mu_cov_hist': mu_cov_hist,
        # 'mu_true': mu1_true.tolist(),
        'policy': pis.tolist()
    }
    return results
