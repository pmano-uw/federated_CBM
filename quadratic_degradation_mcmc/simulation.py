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

from posterior_pymc import update_posterior, update_isolated_posterior, update_EP_posterior, device_posterior_update
from utilities import suppress_stdout_stderr, CholeskyAlgorithm, invwishart_vech_moments

def sim_data(args):
    # Set seed
#     np.random.seed(args['seed'])
    
    # Simulate true beta
    hist_lk_list = [[[]] for n in range(args['N'])]
    diff_hist_lk_list = [[[]] for n in range(args['N'])]
    
    m_counter = np.zeros(args['N']).astype(int)
    k_counter = np.ones(args['N']).astype(int)

    # Sample parameter for each site
    mu1_true = np.random.multivariate_normal(mean=args['mu_true'].flatten(), cov=args['Sigma_true'], size=args['N'])

    for t in range(args['T']):
        for n in range(args['N']):
            # Reassign the variable
            m = m_counter[n]
            k = k_counter[n]
            beta = mu1_true[n]

            # Compute inner product of beta
            beta_term = 0
            for degree in range(args['d']):
                beta_term += beta[degree] * ((k)**(degree)) * (args['delta']**(degree+1))

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

def run_sim(args, sim_round):
    # Print
    print(f"Running sim#{sim_round}")
    d = args['d']
    
    # Intialize empty list
    collab_mean = np.zeros((args['N'], 2, args['T'])); collab_std = np.zeros((args['N'], d, args['T']))
    collab_lap_1_mean = np.zeros((args['N'], 2, args['T'])); collab_lap_1_std = np.zeros((args['N'], d, args['T']))
    collab_lap_2_mean = np.zeros((args['N'], 2, args['T'])); collab_lap_2_std = np.zeros((args['N'], d, args['T']))
    fed_ep_mean = np.zeros((args['N'], 2, args['T'])); fed_ep_std = np.zeros((args['N'], d, args['T']))
    iso_mean = np.zeros((args['N'], 2, args['T'])); iso_std = np.zeros((args['N'], d, args['T']))
    
    # Sim data
    hist_lk, hist_diff_lk, mu_true = sim_data(args)
    # print(hist_diff_lk[0])

    # Set parameters for EP: mean/covariance for the initial Gaussian approximation
    # over phi = [mu, vech(Sigma)]. mu's part matches Cen's Gaussian prior exactly;
    # vech(Sigma)'s part is a moment-matched Gaussian approximation of Cen's
    # InverseWishart(shape_Sigma, nu_Sigma) prior.
    mean_Sigma_vech, cov_Sigma_vech = invwishart_vech_moments(args['shape_Sigma'].item(), args['nu_Sigma'])
    mu_vec = np.concatenate((args['mu_mu'].flatten(), mean_Sigma_vech)).reshape(-1, 1)
    cov_mat = scipy.linalg.block_diag(args['Sigma_mu'], cov_Sigma_vech)

    # Specify the model
    # with suppress_stdout_stderr():
    hybrid_dist_model = CmdStanModel(stan_file="stan_file/hybrid_posterior.stan")
    predictive_model = CmdStanModel(stan_file="stan_file/predictive_posterior.stan")

    # Specify noise
    small_noise = 1
    large_noise = 3

    # Initialize EP state once; carried forward (warm-started) across timesteps below
    # instead of being reset to the flat prior every t.
    r0 = np.linalg.solve(cov_mat, mu_vec)
    Q0 = np.linalg.inv(cov_mat)

    phi_dim = d + d * (d + 1) // 2  # mu (d) + vech(Sigma) (d*(d+1)/2)
    r_list = np.zeros((args['N'], phi_dim))
    Q_list = np.zeros((args['N'], phi_dim, phi_dim))

    r_i = np.linalg.solve(args['site_Sigma'], args['site_mu'])
    Q_i = np.linalg.inv(args['site_Sigma'])

    for n in range(args['N']):
        r_list[n] = r_i.flatten()
        Q_list[n] = Q_i

    r = r0 + np.sum(r_list, axis=0).reshape(-1, 1)
    Q = Q0 + np.sum(Q_list, axis=0)

    # Run sim
    for t in range(5, args['T']):
        hist_diff_lk_flat = []
        hist_diff_lk_lap_1 = []
        hist_diff_lk_lap_2 = []

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
        r_new, Q_new, r_list_new, Q_list_new = update_EP_posterior(r, Q, r_list, Q_list, hist_diff_lk_flat, hybrid_dist_model, args)
        r, Q, r_list, Q_list = r_new, Q_new, r_list_new, Q_list_new
        time5 = time.time()
        beta_mean_ep, beta_std_ep, conv_chain = device_posterior_update(r_new, Q_new, r_list_new, Q_list_new, hist_diff_lk_flat, predictive_model, args)
        time6 = time.time()

        # print(f"Centralize time = {time2-time1:.4f} | Centralize noise time = {time3-time2:.4f} | Iso time = {time4-time3:.4f} | EP inner loop time = {time5-time4:.4f} | EP marginalize = {time6-time5:.4f}")

        # if t%10 == 0:
        # print(f"Sim {sim_round} | time {t} | Iso gap = {np.mean(np.abs(beta_mean_iso - mu_true), axis=0)} | EP gap = {np.mean(np.abs(beta_mean_ep - mu_true), axis=0)} | Took {(time6-time1)/60:.4f} mins")
        # print(f"Sim {sim_round} | time {t} | Iso gap = {np.linalg.norm(beta_mean_iso-mu_true, ord='fro'):.6f} | cent gap = {np.linalg.norm(beta_mean-mu_true, ord='fro'):.6f}")
        print(f"Sim {sim_round} | time {t}  | EP gap = {np.linalg.norm(beta_mean_ep-mu_true, ord='fro'):.4f} | Iso gap = {np.linalg.norm(beta_mean_iso-mu_true, ord='fro'):.4f} | Cen gap = {np.linalg.norm(beta_mean-mu_true, ord='fro'):.4f} | Lap 1 gap = {np.linalg.norm(beta_mean_lap_1-mu_true, ord='fro'):.4f} | Lap 2 gap = {np.linalg.norm(beta_mean_lap_2-mu_true, ord='fro'):.4f} | Took {(time6-time1)/60:.4f} mins")
        # print(f"(Collab) t={t} | mean={beta_mean[:5].flatten()}")
        # print(f"(Isolated) t={t} | mean={beta_mean_iso[:5].flatten()}")
        # print(f"(EP) t={t} | mean={beta_mean_ep[:5].flatten()}")
        # print(f"Ground truth | {mu_true[:5]}")
        # print('-'*20)
        # print(beta_std)
        collab_mean[:, :, t] = np.abs(beta_mean - mu_true); collab_std[:, :, t] = beta_std
        collab_lap_1_mean[:, :, t] = np.abs(beta_mean_lap_1 - mu_true); collab_lap_1_std[:, :, t] = beta_std_lap_1
        collab_lap_2_mean[:, :, t] = np.abs(beta_mean_lap_2 - mu_true); collab_lap_2_std[:, :, t] = beta_std_lap_2
        iso_mean[:, :, t] = np.abs(beta_mean_iso - mu_true); iso_std[:, :, t] = beta_std_iso
        fed_ep_mean[:, :, t] = np.abs(beta_mean_ep - mu_true); fed_ep_std[:, :, t] = beta_std_ep

    # return collab_mean, iso_mean
    return collab_mean.tolist(), \
            collab_lap_1_mean.tolist(), collab_lap_2_mean.tolist(), \
            iso_mean.tolist(), fed_ep_mean.tolist(), \
            collab_std.tolist(), collab_lap_1_std.tolist(), \
            collab_lap_2_std.tolist(), iso_std.tolist(), \
            fed_ep_std.tolist()