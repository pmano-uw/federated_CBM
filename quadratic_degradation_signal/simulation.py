import itertools
import time
import copy
import random

import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

from value_iteration import value_iteration
from posterior import get_prior, update_EP_posterior, update_isolated_posterior, update_posterior, device_posterior_update
from utilities import calc_cost

def sample_transition(beta, args):
    lk_list = np.zeros(args['K'])
    lk_list[0] = args['l_0']
    for t in range(1, args['K']):
        beta_term = 0
        for degree in range(args['d']):
            beta_term += beta[degree] * (t**degree) * (args['delta']**(degree+1))
        lk_list[t] = lk_list[t-1] + beta_term + np.random.randn(1).item() * args['sigma0']
    
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
    state_2 = np.zeros((args['N'], args['M'], args['K']))

    discrete_state_1 = np.zeros((args['N'], args['M'], args['K']))
    discrete_state_2 = np.zeros((args['N'], args['M'], args['K']))

    # Create bins to discretize states
    bins_state_1 = np.linspace(args['l_0'], args['s1_limit'], args['S']-1).ravel()
    bins_state_2 = []
    bins_2d = []
    raw_bins = []

    for k in range(args['K']):
        bins_state_2_k = np.linspace(args['l_0'], args['s1_limit'] * (k+1), args['S']-1).ravel()
        bins_state_2.append(bins_state_2_k)
        
        # Cross-product of all bins
        bins_2d_k = np.stack(list(itertools.product(bins_state_1, bins_state_2_k)))
        bins_2d.append(bins_2d_k)
        raw_bins.append((bins_state_1, bins_state_2_k))
    bins_state_2 = np.stack(bins_state_2)

    mu1_true = np.random.multivariate_normal(mean=args['mu_true'].ravel(), cov=args['Sigma_true'], size=args['N'])

    # Generate true parameters for N sites and M machines
    for n in range(args['N']):
        for m in range(args['M']):
            # Add noise
            if experiment == 'collaborative' and args['lap_noise'] > 0:
                noise = np.random.laplace(scale=args['lap_noise'], size=args['K'])
            else:
                noise = np.zeros(args['K'])

            hist_lk_tensor[n, m, :] = sample_transition(mu1_true[n, :], args)
  
            hist_diff_lk_tensor[n, m, 1:] = np.diff(hist_lk_tensor[n, m, :])
            hist_diff_lk_tensor[n, m, 0] = hist_lk_tensor[n, m, 0]
    
            # Extract states
            state_1 = np.multiply(args['delta'], hist_lk_tensor)
            
            # Discretize states
            discrete_state_1[n, m, :] = discretize_states(state_1[n, m, :], bins_state_1)
            
            # States in 2nd dimension
            state_2[n, m, 0] = hist_diff_lk_tensor[n, m, 0]
            for k in range(1, args['K']):
                state_2[n, m, k] = state_2[n, m, k-1] +  hist_diff_lk_tensor[n, m, k] * k * args['delta']**2
                discrete_state_2[n, m, k] = discretize_states(state_2[n, m, k], bins_state_2[k])
                
    discrete_state_1 = discrete_state_1.astype(int)
    discrete_state_2 = discrete_state_2.astype(int)
    
    # Running posterior update
    m_counter = np.ones(args['N']).astype(int)
    k_counter = np.ones(args['N']).astype(int)

    # Set parameters for EP
    if args['experiment'] == 'EP':
        ## First moment 
        log_norm_mu = np.exp(args['mu_Sigma'] + args['Sigma_Sigma']**2/2)
        mu_vec = np.concatenate((args['mu_mu'].flatten(), log_norm_mu * np.ones(2))).reshape(-1, 1)

        ## Variance
        log_norm_Sigma = (np.exp(args['Sigma_Sigma']**2) - 1) * np.exp(2*args['mu_Sigma'] + args['Sigma_Sigma']**2)
        cov_mat = scipy.linalg.block_diag(args['Sigma_mu'], log_norm_Sigma*np.eye(2))

        # Specify the model
        with suppress_stdout_stderr():
            hybrid_dist_model = CmdStanModel(stan_file="stan_file/hybrid_posterior.stan", model_name=f"hybrid_distribution")
            predictive_model = CmdStanModel(stan_file="stan_file/predictive_posterior.stan", model_name=f"predictive_posterior")

    # Specify noise
    small_noise = 1
    large_noise = 3

    hist_cost_list = [[] for n in range(args['N'])]
    hist_value_gap = [[] for n in range(args['N'])]
    hist_policy_gap = [[] for n in range(args['N'])]
    hist_prob_gap = [[] for n in range(args['N'])]
    
    hist_lk_diff_list = [[[] for m in range(args['M'])] for n in range(args['N'])]
    hist_lk_list = [[[] for m in range(args['M'])] for n in range(args['N'])]

    pi_hist = []
    flag_hist = []
    mu_mean_hist = []
    mu_cov_hist = []
    value_hist = []
    
    values = np.zeros((args['N'], args['K'], args['S']**2))
    pis = np.zeros((args['N'], args['K'], args['S']**2))
    flags = np.zeros((args['N'], args['K'], args['S']**2))
    
    opt_values = np.zeros((args['N'], args['K'], args['S']**2))
    opt_pis = np.zeros((args['N'], args['K'], args['S']**2))
    opt_flags = np.zeros((args['N'], args['K'], args['S']**2))
    opt_probs = np.zeros((args['N'], args['K'], args['S']**2, args['S']**2))
    
    # Derive optimal policy for each site
    # for site in range(args['N']):
    #     betas = true_beta[site, m_counter[site]-1, :].reshape(-1, 1)
    #     opt_values[site, :, :], opt_pis[site, :, :], opt_probs[site, :, :, :], opt_flags[site, :, :] = value_iteration(betas, 0, bins_2d, raw_bins, args, options='optimal')
    #     print(f"Running optimal policy for site {site}")
    for t in range(args['max_T']):
        if args['experiment'] == 'EP':
            r0 = np.linalg.solve(cov_mat, mu_vec)
            Q0 = np.linalg.inv(cov_mat)
            
            # Initialize r_i and Q_i
            r_list = np.zeros((args['N'], 2*d))
            Q_list = np.zeros((args['N'], 2*d, 2*d))

            r_i = np.linalg.solve(args['site_Sigma'], args['site_mu'])
            Q_i = np.linalg.inv(args['site_Sigma'])

            for n in range(args['N']):
                r_list[n] = r_i.flatten()
                Q_list[n] = Q_i

            r = r0 + np.sum(r_list, axis=0).reshape(-1, 1)
            Q = Q0 + np.sum(Q_list, axis=0)

        # print(f"----- Time = {t} -----")
        updated = 0
        # Check termination condition
        if np.all(m_counter == args['M']):
            break
        for i in range(args['N']):  
            # Check update skip condition
            if m_counter[i] == args['M']:
                continue
                
            m = m_counter[i]; m_idx = m - 1; k = k_counter[i] # Variables
            c_state_1 = state_1[i, m_idx, k]; c_state_2 = state_2[i, m_idx, k] # States
            d_state_1 = discrete_state_1[i, m_idx, k]; d_state_2 = discrete_state_2[i, m_idx, k] # Discrete states
            
            # Retrieve pi
            pi = pis[i, k, :].reshape(args['S'], args['S'])
            
            # Diff lk
            diff_lk = hist_diff_lk_tensor[i, m_idx, k]
            lk = hist_lk_tensor[i, m_idx, k]

            # Save history
            hist_lk_diff_list[i][m_idx].append([k, diff_lk.item()])
            hist_lk_list[i][m_idx].append([k, lk.item()])
                    
            # Determine the action from pi (policy) and replacement limit
            if c_state_1 > args['r_limit'] or pi[d_state_1, d_state_2] > 0.5:
                cost_type = 'urgent' if c_state_1 > args['r_limit'] else 'schedule'
                cost = calc_cost(k, args, cost_type)
               
                # Append the cost
                hist_cost_list[i].append([k, cost])
                
                # Increment the counters
                m_counter[i] += 1
                k_counter[i] = 0
                print(f"For sim round {j} | At t = {t} | Machine count: {np.mean(m_counter)} | cost = {cost}")
                
                # Recalculate optimal policies, values, probs
                # betas = true_beta[i, m_counter[i]-1, :].reshape(-1, 1)
                # opt_values[i, :, :], opt_pis[i, :, :], opt_probs[i, :, :, :], opt_flags[i, :, :] = value_iteration(betas, 0, bins_2d, raw_bins, args, options='optimal')
            # else:
            #     print(f"site {i} | true mu1 = {mu1_true[i]} | beta = {true_beta[i, m_idx, :]} | mu1 mean = {mu0_mean[(i+1)*d:(i+2)*d].flatten()}")
        
        if t % args['window'] == 0:
            # print("Updating posterior...")
            # Update posterior
            hist_diff_lk_flat = []
            for n in range(args['N']):
                lk_list = list(itertools.chain.from_iterable(hist_lk_diff_list[n]))
                lk_list = np.stack(lk_list)

                # lk_lap_1[:, 1] += stats.laplace.rvs(scale=small_noise) / t 
                # lk_lap_2[:, 1] += stats.laplace.rvs(scale=large_noise) / t

                hist_diff_lk_flat.append(lk_list)

            if experiment == 'collaborative': 
                mu0_mean, mu0_cov = update_posterior(args, hist_diff_lk_flat, noise)
            elif experiment == 'isolated':
                mu0_mean, mu0_cov = update_isolated_posterior(args, hist_diff_lk_flat)
            elif experiment == 'EP':
                r0, Q0, r_list_new, Q_list_new = update_EP_posterior(r_list, Q_list, r0, Q0, hist_diff_lk_flat, args)
                mu0_mean, mu0_cov = device_posterior_update(r0, Q0, r_list_new, Q_list_new, hist_diff_lk_flat,args)

            mu_mean_hist.append(mu0_mean.tolist())
            mu_cov_hist.append(mu0_cov.tolist())
            
            # print("Running VI...")
            # Run VI for each site
            for site in range(args['N']):
                # print(f"Site {site} |" ,end=' ')
                values[site, :, :], pis[site, :, :], probs, flags[site, :, :] = value_iteration(mu0_mean[i].reshape(-1, 1), mu0_cov[i], bins_2d, raw_bins, args)
                # pi_hist.append(pis[:, :5, :5].tolist()); value_hist.append(values[:5, :5].tolist())

                # Comparing policies, values and probs
                # hist_value_gap[site].append(np.linalg.norm(opt_values[site, :, :] - values[site, :, :], ord='fro').item())
                # hist_policy_gap[site].append(np.linalg.norm(opt_pis[site, :, :] - pis[site, :, :], ord='fro').item())
                
                # Calc Hellinger distance
                # h_dist = 0.5 * np.sum((np.sqrt(np.abs(opt_probs[site, :, :, :])) - np.sqrt(np.abs(probs)))**2, axis=-1)
                # hist_prob_gap[site].append(np.mean(h_dist).item())
            pi_hist.append(pis[0, ::2, :].tolist())
        # Increment k at the end of every sweep
        k_counter += 1 
        
    results = {
        'hist_cost': hist_cost_list,
        # 'hist_value_gap': hist_value_gap,
        # 'hist_policy_gap': hist_policy_gap,
        'mu_mean_hist': mu_mean_hist,
        'mu_cov_hist': mu_cov_hist,
        'mu_true': mu1_true.tolist(),
        'policy': pi_hist
    }
    return results