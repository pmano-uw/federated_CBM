import itertools
import time
import copy
import random
# import pystan
import scipy
from scipy import stats

import numpy as np
import matplotlib.pyplot as plt

from cmdstanpy import CmdStanModel

from posterior import update_posterior, update_isolated_posterior, update_EP_posterior, device_posterior_update, reset_EP_priors
from utilities import suppress_stdout_stderr

def sim_data(args):
    # Set seed
#     np.random.seed(args['seed'])
    
    # Simulate true beta
    hist_lk_list = [[[]] for n in range(args['N'])]
    diff_hist_lk_list = [[[]] for n in range(args['N'])]
    
    m_counter = np.zeros(args['N']).astype(int)
    k_counter = np.ones(args['N']).astype(int)

    # Sample parameter for each site
    mu1_true = np.random.normal(loc=args['mu_true'], scale=args['Sigma_true'], size=args['N'])

    for t in range(args['T']):
        for n in range(args['N']):
            # Reassign the variable
            m = m_counter[n]
            k = k_counter[n]
            beta = mu1_true[n]

            # Compute inner product of beta
            if args['d'] > 1:
                beta_term = 0
                for degree in range(args['d']):
                    beta_term += beta[degree] * ((k)**(degree)) * (args['delta']**(degree+1))
            else:
                beta_term = beta * args['delta']

            # Accumulate lk
            lkm1 = args['l_0'] if k == 1 else hist_lk_list[n][m][-1][1]
            lk = lkm1 + beta_term + np.random.randn(1).item() * args['sigma0'].item()
            diff_lk = lk - lkm1
            
            # Append to the list
            hist_lk_list[n][m].append((k, lk.item()))
            diff_hist_lk_list[n][m].append((k, diff_lk.item()))

            if lk > args['r_limit'] and t != args['T']-1:
                m_counter[n] += 1
                k_counter[n] = 0
            
                hist_lk_list[n].append([])
                diff_hist_lk_list[n].append([])
                
        # Increment k
        k_counter += 1
        
    return hist_lk_list, diff_hist_lk_list, mu1_true

def run_sim_linear(args, sim_round):
    # Print
    print(f"Running sim#{sim_round}")
    d = args['d']
    
    # Intialize empty list
    collab_mean = np.zeros((args['N'], args['T'])); collab_std = np.zeros((args['N'], args['T']))
    collab_lap_1_mean = np.zeros((args['N'], args['T'])); collab_lap_1_std = np.zeros((args['N'], args['T']))
    collab_lap_2_mean = np.zeros((args['N'], args['T'])); collab_lap_2_std = np.zeros((args['N'], args['T']))
    fed_ep_mean = np.zeros((args['N'], args['T'])); fed_ep_std = np.zeros((args['N'], args['T']))
    iso_mean = np.zeros((args['N'], args['T'])); iso_std = np.zeros((args['N'], args['T']))
    
    # Sim data
    hist_lk, hist_diff_lk, mu_true = sim_data(args)
    
    # Set parameters for EP
    mu_Sigma = args['beta_Sigma'] / (args['alpha_Sigma'] - args['d'] - 1)
    ev_initial = np.concatenate((args['mu_mu'].flatten(), mu_Sigma.flatten())).reshape(-1, 1)
    
    # Intialize r and Q
    psi = copy.deepcopy(args['beta_Sigma'])
    nu = copy.deepcopy(args['alpha_Sigma'])
    if d>1:
        cov_Sigma = np.eye(d**2)
        cov_initial = scipy.linalg.block_diag(args['Sigma_mu'], cov_Sigma) 
    else:
        var_Sigma = args['beta_Sigma']**2 / ((nu-1)**2 * (nu-2))
        cov_initial = scipy.linalg.block_diag(args['Sigma_mu']**2, var_Sigma)

    # Specify the model
    # with suppress_stdout_stderr():
    #     hybrid_dist_model = CmdStanModel(stan_file="stan_file/hybrid_posterior.stan", model_name="hybrid_distribution")
    #     predictive_model = CmdStanModel(stan_file="stan_file/predictive_posterior.stan", model_name="predictive_posterior")

    # Specify noise
    small_noise = 1
    large_noise = 3

    # Randomly select adversarial nodes
    if args['adv_node'] > 0:
        all_nodes = list(range(args['N']))
        adv_nodes_list = np.random.choice(all_nodes, size=args['adv_node'], replace=False).tolist()
        args['adv_nodes_list'] = adv_nodes_list
    else:
        args['adv_nodes_list'] = None
    print(args['adv_nodes_list'])

    # Run sim
    for t in range(1, args['T']):
        hist_diff_lk_flat = []
        hist_diff_lk_lap_1 = []
        hist_diff_lk_lap_2 = []

        # Reset EP priors
        r, Q, r_list, Q_list = reset_EP_priors(ev_initial, cov_initial, args)

        for n in range(args['N']):
            lk_list = list(itertools.chain.from_iterable(hist_diff_lk[n]))[:t]
            lk_list = np.stack(lk_list)
            lk_lap_1 = np.copy(lk_list)
            lk_lap_2 = np.copy(lk_list)

            lk_lap_1[:, 1] += stats.laplace.rvs(scale=small_noise) / t
            lk_lap_2[:, 1] += stats.laplace.rvs(scale=large_noise) / t

            hist_diff_lk_flat.append(lk_list)
            hist_diff_lk_lap_1.append(lk_lap_1)
            hist_diff_lk_lap_2.append(lk_lap_2)

        time1 = time.time()
        # Run experiment for centralized model with no noise
        beta_mean, beta_std = update_posterior(args, hist_diff_lk_flat)
        time2 = time.time()

        # Centralized model with noise
        beta_mean_lap_1, beta_std_lap_1 = update_posterior(args, hist_diff_lk_lap_1)
        beta_mean_lap_2, beta_std_lap_2 = update_posterior(args, hist_diff_lk_lap_2)
        time3 = time.time()

        # Isolated model
        beta_mean_iso, beta_std_iso = update_isolated_posterior(args, hist_diff_lk_flat)
        time4 = time.time()

        # Expectation propagation model
        # r, Q, r_list_new, Q_list_new = update_EP_posterior(r, Q, r_list, Q_list, hist_diff_lk_flat, hybrid_dist_model, args)
        # time5 = time.time()
        # beta_mean_ep, beta_std_ep = device_posterior_update(r, Q, r_list_new, Q_list_new, hist_diff_lk_flat, predictive_model, args)
        # time6 = time.time()

        # print(f"Centralize time = {time2-time1:.4f} | Centralize noise time = {time3-time2:.4f} | Iso time = {time4-time3:.4f} | EP inner loop time = {time5-time4:.4f} | EP marginalize = {time6-time5:.4f}")

        # if t%10 == 0:
        # print(f"Sim {sim_round} | time {t} | Iso gap = {np.mean(np.abs(beta_mean_iso.flatten() - mu_true)):.6f} | EP gap = {np.mean(np.abs(beta_mean_ep.flatten() - mu_true)):.6f}")
        # print(f"Sim {sim_round} | time {t} | EP gap = {np.mean(np.abs(beta_mean_ep - mu_true)):.6f} | Iso gap = {np.mean(np.abs(beta_mean_iso.flatten() - mu_true)):.6f} | Cen gap = {np.mean(np.abs(beta_mean.flatten() - mu_true)):.6f} | Lap 1 gap = {np.mean(np.abs(beta_mean_lap_1.flatten() - mu_true)):.6f} | Lap 2 gap = {np.mean(np.abs(beta_mean_lap_2.flatten() - mu_true)):.6f}")
        print(f"Sim {sim_round} | time {t} | Iso gap = {np.mean(np.abs(beta_mean_iso.flatten() - mu_true)):.6f} | Cen gap = {np.mean(np.abs(beta_mean.flatten() - mu_true)):.6f} | Lap 1 gap = {np.mean(np.abs(beta_mean_lap_1.flatten() - mu_true)):.6f} | Lap 2 gap = {np.mean(np.abs(beta_mean_lap_2.flatten() - mu_true)):.6f}")

        # Centralized model with noise sigma = 1
        collab_mean[:, t] = np.abs(beta_mean.flatten() - mu_true); collab_std[:, t] = beta_std.flatten()
        collab_lap_1_mean[:, t] = np.abs(beta_mean_lap_1.flatten() - mu_true); collab_lap_1_std[:, t] = beta_std_lap_1.flatten()
        collab_lap_2_mean[:, t] = np.abs(beta_mean_lap_2.flatten() - mu_true); collab_lap_2_std[:, t] = beta_std_lap_2.flatten()
        iso_mean[:, t] = np.abs(beta_mean_iso.flatten() - mu_true); iso_std[:, t] = beta_std_iso.flatten()
        # fed_ep_mean[:, t] = np.abs(beta_mean_ep - mu_true); fed_ep_std[:, t] = beta_std_ep.flatten()

        # if np.mean(np.abs(beta_mean_iso.flatten() - mu_true)) <= np.mean(np.abs(beta_mean_ep - mu_true)):
        #     print(beta_mean_ep)

    # return collab_mean, iso_mean
    return {"collab": collab_mean.tolist(), "lap_1": collab_lap_1_mean.tolist(), 
            "lap_2": collab_lap_2_mean.tolist(), "iso": iso_mean.tolist(), 
            "fed_ep": fed_ep_mean.tolist(), "collab_std": collab_std.tolist(), 
            "lap_1_std": collab_lap_1_std.tolist(), "lap_2_std": collab_lap_2_std.tolist(), 
            "iso_std": iso_std.tolist(), "fed_ep_std": fed_ep_std.tolist(),
            "adv_nodes": args['adv_nodes_list']}