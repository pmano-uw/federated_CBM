import numpy as np
import scipy
import itertools
import copy
import os
import time
import pandas as pd

import multiprocessing as mp
from tqdm import tqdm
from joblib import Parallel, delayed
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import acf

from utilities import suppress_stdout_stderr, check_symmetric, CholeskyAlgorithm, within_chain_var, between_chain_var, autocorr_estimate

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
    max_gap = 100000
    best_r = np.copy(r); best_Q = np.copy(Q); best_r_list = np.copy(r_list); best_Q_list = np.copy(Q_list)

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
            Sigma_cavity = CholeskyAlgorithm(np.linalg.inv(Q_cavity), args['epsilon']) + args['epsilon'] * np.eye(4)
            # print(f"C {c} Site {i} | mu = {mu_cavity.flatten()}, sigma = {Sigma_cavity.flatten()}")
            
            if not check_symmetric(Sigma_cavity):
                Sigma_cavity = (Sigma_cavity + Sigma_cavity.T) / 2
            
            # Use stan
            dat = {
                "N": len(lk_list), "Delta": args['delta'], "sigma_0": args['sigma0'].item(),
                "Delta_l_k": lk_list.tolist(), "k": k_list.tolist(), "mu_i": mu_cavity.flatten(), "Sigma_i": Sigma_cavity
            }
            with suppress_stdout_stderr():
                fit = model.sample(data=dat, iter_sampling=3000, iter_warmup=1000, chains=4, show_progress=False)

            df_fit = fit.draws_pd(inc_warmup=False)
            hybrid_samples = df_fit[[f'phi[{x}]' for x in range(1, 5)]].values
            # print(fit.diagnose())
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
        # print(f"mu gap = {np.linalg.norm(mu - mu_old)}")
        # print('-'*30)
        if np.linalg.norm(mu - mu_old) < args['ep_tol']:
            return r, Q, r_list, Q_list
        # elif np.linalg.norm(mu - mu_old) < max_gap:
        #     max_gap = np.linalg.norm(mu - mu_old)
        #     best_r = np.copy(r); best_Q = np.copy(Q); best_r_list = np.copy(r_list); best_Q_list = np.copy(Q_list)

    return r, Q, r_list, Q_list

def device_posterior_update(r, Q, r_list, Q_list, diff_lk_list, model, args):    
    d = args['d']; N = args['N']
    mu_list = np.zeros((N, d))
    sigma_list = np.zeros((N, d))
    conv_chain_percs = []

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
        Sigma_cavity = CholeskyAlgorithm(np.linalg.inv(Q_cavity), args['epsilon']) + args['epsilon'] * np.eye(4)
        # print(np.linalg.cond(Sigma_cavity))
        # print(f"Site {i+1} | {mu_cavity.flatten()} ")
        # print(Sigma_cavity)

        if not check_symmetric(Sigma_cavity):
            Sigma_cavity = (Sigma_cavity + Sigma_cavity.T) / 2

        dat = {
            "N": len(lk_list),
            "Delta": args['delta'],
            "sigma_0": args['sigma0'].item(),
            "Delta_l_k": lk_list,
            "k": k_list,
            "mu_i": mu_cavity.flatten(),
            "Sigma_i": Sigma_cavity
        }
        # treedepth = 10 if len(k_list) > 10 else 15
        for idx in range(args['max_iter']):
            mix_flag = 0
            conv_chain_perc = 0

            with suppress_stdout_stderr():
                fit_pred = model.sample(data=dat, iter_sampling=args['num_samples'], iter_warmup=args['num_burnin'], chains=args['num_chains'], adapt_delta=0.99, inits=0.5,
                                            show_progress=False, save_warmup=False)

            df_fit_pred = fit_pred.draws(inc_warmup=False)
            col_name = fit_pred.draws_pd(inc_warmup=False).columns
            col_idx = [np.where(col_name=='beta[1]')[0][0], np.where(col_name=='beta[2]')[0][0]]
            warmup_idx = args['num_burnin']
            beta_samples = df_fit_pred[:, :, col_idx]

            # Check if there are any converging chains
            M = beta_samples.shape[1]
            rho_hat_m = np.zeros((args['max_lag'], M, 2))
            for m in range(M):
                for j in range(2):
                    with suppress_stdout_stderr():
                        autocorr_j = acf(beta_samples[:, m, j], nlags=args['max_lag'], fft=True)[1:]
                    rho_hat_m[:, m, j] = autocorr_j

            # Check if there are any mixing chains
            for j in range(2):
                tmp_mix_idx = np.argwhere(np.mean(np.abs(rho_hat_m[:,:,j]), axis=0) < args['rho_threshold']).ravel().tolist()
                print(tmp_mix_idx)
                if len(tmp_mix_idx) > 1:
                    mix_flag += 1
                
            # Escape criteria
            if mix_flag >= 2:
                break
        
        # Calculate sample variance of the chains
        N = beta_samples.shape[0]
        M = beta_samples.shape[1]

        # Compute autorrelation
        rho_hat, R_hat, mix_chains_idx, _ = autocorr_estimate(beta_samples[warmup_idx:, :, :], args['max_lag'] ,args)
        tau = (1 + 2* np.sum(rho_hat, axis=0)).astype(int)
        tau = np.where(tau>0, tau, 1)

        beta_mean = np.zeros(2)
        beta_std = np.zeros(2)
        for j in range(2):
            samples_j = beta_samples[warmup_idx::tau[j], mix_chains_idx[j], j].ravel()
            beta_mean[j] = np.mean(samples_j)
            beta_std[j] = np.std(samples_j)

            conv_chain_perc += len(mix_chains_idx[j]) / args['num_chains']
        conv_chain_percs.append(conv_chain_perc/2)
        # print(f"tau = {tau} | beta = {beta_mean} | R_hat = {R_hat} | ESS = {np.floor(N / tau)}")

        # Plot chains
        fig, ax = plt.subplots(2, 2)
        for j in range(2):
            ax[j, 0].plot(beta_samples[:, :, j])
            ax[j, 1].plot(beta_samples[warmup_idx::tau[j], mix_chains_idx[j], j])
        fig.savefig('figure/chain.png')
        plt.close(fig)

        # beta_mean = np.mean(beta_samples, axis=0)
        # beta_std = np.std(beta_samples, axis=0)
        # print(beta_mean)
        # print(np.mean(df_fit_pred[['mu[1]', 'mu[2]']].values, axis=0))
        # print('-'*30)

        mu_list[i] = beta_mean
        sigma_list[i] = beta_std

    conv_chain_percs = np.mean(conv_chain_percs)

    return mu_list, sigma_list, conv_chain_percs