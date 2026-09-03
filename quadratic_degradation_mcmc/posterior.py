import numpy as np
import scipy
import itertools
import copy
import os
import time
import pandas as pd

# import pystan
import multiprocessing as mp
from tqdm import tqdm
from joblib import Parallel, delayed
# from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
from scipy import stats

from utilities import suppress_stdout_stderr, check_symmetric, CholeskyAlgorithm

def get_mu_sample(args, mu, Sigma, betas):
    precision = np.linalg.inv(args['Sigma_mu']) + args['N'] * np.linalg.inv(Sigma)
    linear_shift = np.linalg.solve(args['Sigma_mu'], args['mu_mu']) +  np.linalg.solve(Sigma, np.sum(betas, axis=0).reshape(-1, 1))  

    mean = np.linalg.solve(precision, linear_shift)
    cov = np.linalg.inv(precision)
    
    mu_sample = stats.multivariate_normal.rvs(mean=mean.flatten(), cov=cov)
    return mu_sample

def get_Sigma_sample(args, mu, Sigma, betas):
    betas_central = betas - mu
    alpha_param = args['shape_Sigma'] + args['N']
    nu_param = args['nu_Sigma'] + betas_central.T @ betas_central
    
    Sigma_sample = stats.invwishart.rvs(df=alpha_param.item(), scale=nu_param)
    return Sigma_sample

def get_betas_sample(args, mu, Sigma, betas, diff_lk_array):
    # Initialize empty betas
    d = args['d']
    new_betas = np.zeros((args['N'], args['d']))
    Sigma_inv = np.linalg.inv(Sigma)

    for n in range(args['N']):
        lk_list_n = np.stack(diff_lk_array[n])
        tt_T = np.zeros((len(diff_lk_array[n]), d))
        for idx in range(d):
            tt_T[:, idx] = (lk_list_n[:, 0] ** idx) * (args['delta']**(idx+1))

        new_diff_lk = lk_list_n[:, 1].reshape(-1, 1)
        A = Sigma_inv + tt_T.T @ tt_T / (args['sigma0']**2)
        b = (np.dot(Sigma_inv, mu) + np.sum(np.multiply(new_diff_lk, tt_T), axis=0) / (args['sigma0']**2)).reshape(-1, 1)
        mu0_mean = np.linalg.solve(A, b).flatten()
        mu0_cov = np.linalg.inv(A)
        new_betas[n] = stats.multivariate_normal.rvs(mean=mu0_mean.flatten(), cov=mu0_cov)

    return new_betas

def update_posterior(args, diff_lk_input):
    diff_lk_array = np.stack(diff_lk_input)
    
    # Initialize memories
    mu_hist = np.zeros((args['n_chains'], args['gibbs_T'], args['d']))
    Sigma_hist = np.zeros((args['n_chains'], args['gibbs_T'], args['d'], args['d']))
    betas_hist = np.zeros((args['N'], args['n_chains'], args['gibbs_T'], args['d']))

    # Run Gibbs
    for chain in range(args['n_chains']):
        # Randomly sample the starting point according to the prior
        mu = stats.multivariate_normal.rvs(mean=args['mu_mu'].flatten(), cov=args['Sigma_mu'])
        Sigma = stats.invwishart.rvs(df=args['shape_Sigma'].item(), scale=args['nu_Sigma'])
        betas = stats.multivariate_normal.rvs(mean=mu, cov=Sigma, size=args['N'])
        
        for t in range(args['gibbs_T']):      
            # Get a sample from mu
            new_mu = get_mu_sample(args, mu, Sigma, betas)

            # Get a sample from Sigma
            new_Sigma = get_Sigma_sample(args, new_mu, Sigma, betas)

            # Get samples for betas
            new_betas = get_betas_sample(args, new_mu, new_Sigma, betas, diff_lk_array)

            # Update params
            mu = new_mu; Sigma = new_Sigma; betas = new_betas
            
            # Append history
            mu_hist[chain, t] = mu
            Sigma_hist[chain, t] = Sigma
            betas_hist[:, chain, t] = betas
    
    # Take only the samples after the warm-up period
    mu_hist = mu_hist[:, args['warm_up']:, :]
    Sigma_hist = Sigma_hist[:, args['warm_up']:, :]
    betas_hist = betas_hist[:, :, args['warm_up']:, :]

    # Collapse the array
    mu_hist = mu_hist.reshape(-1, args['d'])
    Sigma_hist = Sigma_hist.reshape(-1, args['d'], args['d'])
    betas_hist = betas_hist.reshape(args['N'], -1, args['d'])
    
    mean_betas = np.mean(betas_hist, axis=1)
    std_betas = np.std(betas_hist, axis=1)

    return mean_betas, std_betas

def update_isolated_posterior(args, diff_lk_input):
    d = args['d']
    mu0_mean = np.zeros((args['N'], args['d']))
    mu0_std = np.zeros((args['N'], args['d']))
    
    for n in range(args['N']):
        Sigma_inv = np.linalg.inv(args['Sigma_mu'])
        
        lk_list_n = np.stack(diff_lk_input[n])
        tt_T = np.zeros((len(diff_lk_input[n]), d))
        for idx in range(d):
            tt_T[:, idx] = (lk_list_n[:, 0] ** idx) * (args['delta']**(idx+1))

        new_diff_lk = lk_list_n[:, 1].reshape(-1, 1)
        A = Sigma_inv + tt_T.T @ tt_T / (args['sigma0']**2)
        b = np.dot(Sigma_inv, args['mu_mu']) + (np.sum(np.multiply(new_diff_lk, tt_T), axis=0) / (args['sigma0']**2)).reshape(-1, 1)
        mu0_mean[n] = np.linalg.solve(A, b).flatten()
        mu0_cov =  np.linalg.inv(A)

        mu0_std[n] = np.sqrt(np.diag(mu0_cov))
    return mu0_mean, mu0_std

def update_EP_posterior(r, Q, r_list, Q_list, diff_lk_input, model, args):
    d = args['d']
    for c in range(args['C']):
        # Device-side
        r_delta = 0
        Q_delta = 0

        r_list_new = np.zeros((args['N'], 2*d))
        Q_list_new = np.zeros((args['N'], 2*d, 2*d))

        for i in range(args['N']):
            # Stack data
            lk_list_n = np.stack(diff_lk_input[i])
            k_list = lk_list_n[:, 0]
            lk_list = lk_list_n[:, 1]

            # Subtract old r from r0 and old Q from Q0 to remove the impact of old params
            r_cavity = r - r_list[i].reshape(-1, 1)
            Q_cavity = Q - Q_list[i]
            
            # Run MCMC
            mu_cavity = np.linalg.solve(Q_cavity, r_cavity)
            Sigma_cavity = CholeskyAlgorithm(np.linalg.inv(Q_cavity), args['epsilon'])
            # print(f"C {c} Site {i} | mu = {mu_cavity.flatten()}, sigma = {Sigma_cavity.flatten()}")
            
            if not check_symmetric(Sigma_cavity):
                Sigma_cavity = (Sigma_cavity + Sigma_cavity.T) / 2

            def proposal_distribution(x):
                # Generate concatenated samples from proposal distribution which is a mixture of normal-halfcauchy-normal
                big_cov = scipy.linalg.block_diag(args['Sigma_mu'], args['scale_Sigma'], args['scale_Sigma'])
                x_candidate = stats.multivariate_normal.rvs(mean=x, cov=big_cov)
                return x_candidate

            def log_target_distribution(x, mu_cav, cov_cav, lk_arr, k_arr):
                # Denote variables
                mu = x[:2]
                cov_tau = np.diag(x[2:])
                # Log of target distribution
                log_phi_pdf = stats.multivariate_normal.logpdf(x.flatten(), mean=mu_cav.flatten(), cov=cov_cav)
                log_mu_pdf = 0
                for j in range(len(lk_arr)):
                    tk = np.array([args['delta'], args['delta']**2 * k_arr[j]])
                    tk_col = tk.reshape(-1, 1)
                    mu_tk = np.inner(mu, tk).item()
                    sigma_tk = tk_col.T @ cov_tau @ tk
                    log_mu_pdf += stats.norm.logpdf(lk_arr[j], loc=mu_tk, scale=np.sqrt(sigma_tk))

                return log_phi_pdf + log_mu_pdf

            x0 = np.concatenate((args['mu_mu'].ravel(), args['shape_Sigma'], args['shape_Sigma']))
            xt = np.copy(x0) # Set the current candidate to starting point
            hybrid_samples = np.zeros((args['num_samples'], 4))
            for idx in range(args['num_samples']):
                xt_candidate = proposal_distribution(xt)
                if np.any(xt_candidate < 0):
                    hybrid_samples[idx, :] = xt
                    continue
                log_acceptance_ratio = log_target_distribution(xt_candidate, mu_cavity, Sigma_cavity, lk_list, k_list) - log_target_distribution(xt, mu_cavity, Sigma_cavity, lk_list, k_list)

                if np.log(np.random.rand()) < log_acceptance_ratio.item():
                    # print(log_target_distribution(xt_candidate, mu_cavity, Sigma_cavity, lk_list, k_list))
                    # print(log_target_distribution(xt, mu_cavity, Sigma_cavity, lk_list, k_list))
                    # print(f'{idx} | data = {lk_list[0]} | old estimate = {np.sum(xt[:2])} | new estimate = {np.sum(xt_candidate[:2])} | log pdf = {log_acceptance_ratio}')
                    xt = xt_candidate
                hybrid_samples[idx, :] = xt

            # Extract r_hybrid and Q_hybrid
            mu_hybrid = np.mean(hybrid_samples, axis=0).reshape(-1, 1)
            cov_hybrid = np.cov(hybrid_samples.T)
            # print(f"C {c} Site {i} | mu = {mu_hybrid.flatten()}")
            
            r_hybrid = np.linalg.solve(cov_hybrid, mu_hybrid)
            Q_hybrid = np.linalg.inv(cov_hybrid)

            # Update the contribution to the central prior
            r_delta += r_hybrid - r
            Q_delta += Q_hybrid - Q
            # print(f"C {c} Site {i} | delta = {cov_hybrid}")

            # Append new r_list and Q_list (local approximation)
            r_list_new[i] = (r_hybrid - r_cavity).flatten()
            Q_list_new[i] = Q_hybrid - Q_cavity

        # Update global approximation
        r_new = r + r_delta
        Q_new = Q + Q_delta

        # Update parameters
        r_old = r; Q_old = Q
        r = r_new; Q = Q_new

        r_list = copy.deepcopy(r_list_new)
        Q_list = copy.deepcopy(Q_list_new)

        # Check for convergence
        mu_old = np.linalg.solve(Q_old, r_old)
        mu = np.linalg.solve(Q, r)
        
        # print(f"mu = {mu}")
        # print(f"Communication round: {c}")
        # print(f"r = {r.flatten()}")
        print(f"mu gap = {np.linalg.norm(mu - mu_old)}")
        # print('-'*30)
        if np.linalg.norm(mu - mu_old) < args['ep_tol']:
            return r, Q, r_list, Q_list

    return r, Q, r_list, Q_list

def device_posterior_update(r, Q, r_list, Q_list, diff_lk_list, model, args):    
    d = args['d']; N = args['N']
    mu_list = np.zeros((N, d))
    sigma_list = np.zeros((N, d))

    for i in range(N):
        # Stack data
        lk_list_n = np.stack(diff_lk_list[i])
        
        k_list = lk_list_n[:, 0]
        lk_list = lk_list_n[:, 1]
        # Retrieve cavity dist.
        r_cavity = r - r_list[i].reshape(-1, 1)
        Q_cavity = Q - Q_list[i]

        # Run MCMC
        mu_cavity = np.linalg.solve(Q_cavity, r_cavity)
        Sigma_cavity = CholeskyAlgorithm(np.linalg.inv(Q_cavity), args['epsilon'])

        def proposal_distribution(x):
            # Generate concatenated samples from proposal distribution which is a mixture of normal-halfcauchy-normal
            big_cov = scipy.linalg.block_diag(args['Sigma_mu'], args['Sigma_mu'], args['scale_Sigma'], args['scale_Sigma'])
            x_candidate = stats.multivariate_normal.rvs(mean=x, cov=big_cov)
            return x_candidate

        def log_target_distribution(x, mu_cav, cov_cav, lk_arr, k_arr):
            # Denote variables
            beta = x[:2].reshape(-1, 1); cov_tau = np.diag(x[4:])
            # Log of target distribution
            log_phi_pdf = stats.multivariate_normal.logpdf(x[2:], mean=mu_cav.flatten(), cov=cov_cav)
            log_beta_pdf = stats.multivariate_normal.logpdf(x[:2], mean=x[2:4], cov=cov_tau)
            tk = np.zeros((len(lk_arr), 2))
            tk[:, 0] = args['delta']
            tk[:, 1] = args['delta']**2 * k_arr
            beta_tk_list = (tk @ beta).flatten()
            log_lk_pdf_list = stats.norm.logpdf(lk_arr, loc=beta_tk_list, scale=args['sigma0'])
            log_lk_pdf = np.sum(log_lk_pdf_list)

            # print(log_phi_pdf)
            # print(log_beta_pdf)
            # print(log_lk_pdf)
            # print('-'*30)
            return log_phi_pdf + log_beta_pdf + log_lk_pdf

        # Starting point
        x0 = np.concatenate((args['mu_mu'].ravel(), args['mu_mu'].ravel(), args['shape_Sigma'], args['shape_Sigma']*0.1))
        xt = np.copy(x0) # Set the current candidate to starting point
        beta_samples = np.zeros((args['num_samples'], 2))
        for idx in range(args['num_samples']):
            xt_candidate = proposal_distribution(xt)
            if np.any(xt_candidate < 0):
                beta_samples[idx, :] = xt[:2]
                continue
            log_acceptance_ratio = log_target_distribution(xt_candidate, mu_cavity, Sigma_cavity, lk_list, k_list) - log_target_distribution(xt, mu_cavity, Sigma_cavity, lk_list, k_list)

            if np.log(np.random.rand()) < log_acceptance_ratio.item():
                # print(log_target_distribution(xt_candidate, mu_cavity, Sigma_cavity, lk_list, k_list))
                # print(log_target_distribution(xt, mu_cavity, Sigma_cavity, lk_list, k_list))
                # print(f'{idx} | data = {lk_list[0]} | old estimate = {np.sum(xt[:2])} | new estimate = {np.sum(xt_candidate[:2])} | log pdf = {log_acceptance_ratio}')
                xt = xt_candidate
            beta_samples[idx, :] = xt[:2]
        # print(beta_samples[args['burn_in']:, :])
        beta_mean = np.mean(beta_samples[args['num_burnin']:, :], axis=0)
        beta_std = np.std(beta_samples[args['num_burnin']:, :], axis=0)
        print(beta_mean)

        mu_list[i] = beta_mean
        sigma_list[i] = beta_std

    return mu_list, sigma_list

        # def log_proposal_density(x, y):
        #     # Log probability density of proposing y from x
        #     beta_x = x[:2]; mu_x = x[2:4]; tau1_x = x[4]; tau2_x = x[5]
        #     beta_y = y[:2]; mu_y = y[2:4]; tau1_y = y[4]; tau2_y = y[5]

        #     log_beta_pdf = stats.multivariate_normal.logpdf(beta_x, mean=beta_y, cov=args['Sigma_mu'])
        #     log_mu_pdf = stats.multivariate_normal.logpdf(mu_x, mean=mu_y, cov=args['Sigma_mu'])
        #     log_tau1_pdf = stats.norm.logpdf(tau1_x, loc=tau1_y, scale=args['scale_Sigma'])
        #     log_tau2_pdf = stats.norm.logpdf(tau2_x, loc=tau2_y, scale=args['scale_Sigma'])
        #     return log_beta_pdf + log_mu_pdf + log_tau1_pdf + log_tau2_pdf

        # beta_i = stats.multivariate_normal.rvs(mean=x[:2], cov=args['Sigma_mu']).ravel()
        #     mu = stats.multivariate_normal.rvs(mean=x[2:4], cov=args['Sigma_mu']).ravel()
        #     tau1 = stats.norm.rvs(loc=x[4], scale=args['scale_Sigma'])
        #     tau2 = stats.norm.rvs(loc=x[5], scale=args['scale_Sigma'])

        #     x_candidate = np.zeros(6)
        #     # Concatenate samples
        #     x_candidate[:2] = beta_i
        #     x_candidate[2:4] = mu
        #     x_candidate[4] = tau1
        #     x_candidate[5] = tau2