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
from utilities import calc_cost, suppress_stdout_stderr

def sample_transition(beta, args):
    while True:
        lk_list = np.zeros(args['K'])
        lk_list[0] = args['l_0']
        for t in range(1, args['K']):
            lk_list[t] = lk_list[t-1] + beta*args['delta'] + np.random.randn(1).item() * args['sigma0']
        if np.any(lk_list >= args['r_limit']) and np.all(lk_list >= 0) and lk_list[1] < args['r_limit']:
            return lk_list

def discretize_states(lk_list, bins):
    discrete_state_space = np.digitize(lk_list, bins, right=False)
    return discrete_state_space

def simulation(args, j, experiment='collaborative'):
    print(f"Running simulation #{j}")
    d = args['d']
    N = args['N']
    
    hist_lk_tensor = np.zeros((args['N'], args['M'], args['K']))
    hist_diff_lk_tensor = np.zeros((args['N'], args['M'], args['K']))
    hist_cost_tensor = np.zeros((args['N'], args['M'], args['K']))
    discrete_state_1 = np.zeros((args['N'], args['M'], args['K']))

    # Create bins to discretize states
    bins_state_1 = np.linspace(args['l_0'], args['s1_limit'], args['S']-1).flatten()
    
    # Simulate the parameter for each site
    mu1_true = stats.norm.rvs(loc=args['mu_true'].item(), scale=args['Sigma_true'], size=args['N'], random_state=j+1234)
    # print(mu1_true)

    # Generate true parameters for N sites and M machines
    for n in range(args['N']):
        for m in range(args['M']):
            # Add noise
            if experiment == 'collaborative' and args['lap_noise'] > 0:
                noise = np.random.laplace(scale=args['lap_noise'], size=args['K'])
            else:
                noise = np.zeros(args['K'])

            hist_lk_tensor[n, m, :] = sample_transition(mu1_true[n], args) 
            hist_diff_lk_tensor[n, m, 1:] = np.diff(hist_lk_tensor[n, m, :]) 
            hist_diff_lk_tensor[n, m, 0] = hist_lk_tensor[n, m, 0]

            # Extract states
            state_1 = args['delta'] * hist_lk_tensor

            # Discretize states
            discrete_state_1[n, m, :] = discretize_states(state_1[n, m, :], bins_state_1)
    # print("Finish generating data")
    discrete_state_1 = discrete_state_1.astype(int)
    # Running posterior update
    m_counter = np.ones(args['N']).astype(int)
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
    opt_probs = np.zeros((args['N'], args['K'], args['S'], args['S']))
    
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

    new_bins = copy.deepcopy(bins_state_1)
    new_bins[1:] = bins_state_1[1:] - np.diff(bins_state_1)[0] / 2
    new_bins = np.append(new_bins, args['s1_limit'])

    # Calculate optimal transition prob following true beta
    for site in range(args['N']):
        opt_probs[site] = calc_transition_prob_opt(new_bins, bins_state_1, mu1_true[site], args)

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
            if c_state_1 > args['r_limit'] or pi[d_state_1] > 0.5 or k_counter[i] == args['K']-1:
                cost_type = 'urgent' if c_state_1 > args['r_limit'] else 'schedule'
                cost = calc_cost(k, args, cost_type)
               
                # Append the cost
                hist_cost_list[i].append([k, cost])
                
                # Increment the counters
                m_counter[i] += 1
                k_counter[i] = 0
                print(f"Round {j} | t = {t} | k = {k} | site = {i} | Count: {m_counter[:10]} | cost = {cost:.6f} | {c_state_1:.4f}")
                # print(np.mean(np.abs(mu0_mean - mu1_true)))
                # print('-'*30)
                
        if t % args['window'] == 0:
            # print("Updating posterior...")
            # Update posterior
            if experiment == 'collaborative': 
                mu0_mean, mu0_cov = update_posterior(args, hist_lk_diff_list, noise)
                # mu0_mean_iso, mu0_cov_iso = update_isolated_posterior(args, hist_lk_diff_list)
                # gap_collab = np.linalg.norm(mu1_true - mu0_mean.flatten())
                # gap_iso = np.linalg.norm(mu1_true - mu0_mean_iso.flatten())
            elif experiment == 'isolated':
                # print('Updating posterior')
                mu0_mean, mu0_cov = update_isolated_posterior(args, hist_lk_diff_list)
            elif experiment == 'EP':
                r, Q, r_list, Q_list = reset_EP_priors(ev_initial, cov_initial, args)
                r, Q, r_list_new, Q_list_new = update_EP_posterior(r, Q, r_list, Q_list, hist_lk_diff_list, hybrid_dist_model, args)
                mu0_mean, mu0_cov = device_posterior_update(r, Q, r_list_new, Q_list_new, hist_lk_diff_list, predictive_model, args)

            mu_mean_hist.append(mu0_mean.tolist())
            mu_cov_hist.append(mu0_cov.tolist())
            # gap = np.linalg.norm(mu1_true - mu0_mean.flatten())
            # print(f"t = {t} | collab gap = {gap_collab}, iso gap = {gap_iso}")
            # print("Running VI...")
            # Run VI for each site
            for site in range(args['N']):
                values[site], pis[site], probs = value_iteration(mu0_mean[site], mu0_cov[site], bins_state_1, args)
                values_opt = VI_opt(pis[site], opt_probs[site], new_bins, args) # Evaluate pi from estimates using true beta transition
                values_under_opt_trans[site].append(values_opt[0, 0])
                # Comparing policies, values and probs
                # hist_value_gap[site].append(np.linalg.norm(values_opt - values[site], ord='fro').item())
                # hist_policy_gap[site].append(np.linalg.norm(opt_pis[site, :, :] - pis[site, :, :], ord='fro').item())

                # Calc Hellinger distance
                # h_dist = 0.5 * np.sum((np.sqrt(np.abs(opt_probs[site, :, :, :])) - np.sqrt(np.abs(probs)))**2, axis=-1)
                # hist_prob_gap[site].append(np.mean(h_dist).item())
            # print(hist_value_gap[0])
        # Increment k at the end of every sweep
        k_counter += 1 
    values_under_opt_trans = np.stack(values_under_opt_trans)
    results = {
        'hist_cost': hist_cost_list,
        'values_opt_prob': values_under_opt_trans.tolist(),
        # 'hist_value_gap': hist_value_gap,
        # 'hist_policy_gap': hist_policy_gap,
        'mu_mean_hist': mu_mean_hist,
        'mu_cov_hist': mu_cov_hist,
        'mu_true': mu1_true.tolist(),
        'policy': pis.tolist()
    }
    return results