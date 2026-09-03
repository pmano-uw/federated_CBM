import itertools
import copy
import os
import tempfile
import time
import pandas as pd
from functools import partial
# import arviz as az

# import pystan
import multiprocessing as mp
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from scipy import stats
# import pymc as pm
# import pytensor.tensor as pt
import numpy as np
import scipy

from utilities import suppress_stdout_stderr, CholeskyAlgorithm, ensure_positive_definite, within_chain_var, between_chain_var, autocorr_estimate

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

def _ep_site_update(i, r, Q, r_list, Q_list, diff_lk_input, model, args, phi_dim):
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

    # Always symmetrize (cheap, always safe): Stan's own symmetry check is much
    # stricter than check_symmetric's tolerance, so a matrix that looks "close enough"
    # here can still be rejected by multi_normal_lpdf on the Stan side.
    Sigma_cavity = (Sigma_cavity + Sigma_cavity.T) / 2

    # Use stan
    dat = {
        "N": len(lk_list), "Delta": args['delta'], "sigma_0": args['sigma0'].item(),
        "Delta_l_k": lk_list.tolist(), "k": k_list.tolist(), "mu_i": mu_cavity.flatten(), "Sigma_i": Sigma_cavity,
        "mu_mu": args['mu_mu'].flatten().tolist(), "Sigma_mu": args['Sigma_mu'].tolist(),
        "shape_Sigma": args['shape_Sigma'].item(), "nu_Sigma": args['nu_Sigma'].tolist()
    }
    with tempfile.TemporaryDirectory() as tmp_out, suppress_stdout_stderr():
        fit = model.sample(data=dat, iter_sampling=3000, iter_warmup=1500, chains=4, show_progress=False, output_dir=tmp_out)
        df_fit = fit.draws_pd(inc_warmup=False)

    hybrid_samples = df_fit[[f'phi[{x}]' for x in range(1, phi_dim + 1)]].values

    mu_hybrid = np.mean(hybrid_samples, axis=0).reshape(-1, 1)
    cov_hybrid = np.cov(hybrid_samples.T)

    r_hybrid = np.linalg.solve(cov_hybrid, mu_hybrid)
    Q_hybrid = np.linalg.inv(cov_hybrid)

    # Damped site update (Vehtari et al. 2020, EP as a Way of Life, p.17):
    # new_site_factor = old_site_factor + damping * (undamped_new_site_factor - old_site_factor),
    # blended in natural-parameter space. damping=1 recovers the undamped update.
    r_factor_new = r_hybrid - r_cavity
    Q_factor_new = Q_hybrid - Q_cavity

    r_list_new_i = (r_list[i].reshape(-1, 1) + args['damping'] * (r_factor_new - r_list[i].reshape(-1, 1))).flatten()
    Q_list_new_i = Q_list[i] + args['damping'] * (Q_factor_new - Q_list[i])

    # Positive-definiteness guarantee (Vehtari et al. 2020, sec 5.3): keeping every
    # site's own precision PD is sufficient to guarantee the cavity and global
    # precision stay PD too, since both are sums of PD matrices.
    Q_list_new_i = ensure_positive_definite(Q_list_new_i)

    return r_list_new_i, Q_list_new_i

def update_EP_posterior(r, Q, r_list, Q_list, diff_lk_input, model, args):
    d = args['d']
    phi_dim = d + d * (d + 1) // 2  # mu (d) + vech(Sigma) (d*(d+1)/2)
    max_gap = 100000
    best_r = np.copy(r); best_Q = np.copy(Q); best_r_list = np.copy(r_list); best_Q_list = np.copy(Q_list)

    for c in range(args['C']):
        # Device-side
        r_delta = 0
        Q_delta = 0

        r_list_new = np.zeros((args['N'], phi_dim))
        Q_list_new = np.zeros((args['N'], phi_dim, phi_dim))

        # Sites are independent within a round (all read the same round-starting r, Q),
        # so they can be computed concurrently.
        site_results = Parallel(n_jobs=args.get('site_n_jobs', 1))(
            delayed(_ep_site_update)(i, r, Q, r_list, Q_list, diff_lk_input, model, args, phi_dim)
            for i in range(args['N'])
        )
        for i, (r_list_new_i, Q_list_new_i) in enumerate(site_results):
            r_list_new[i] = r_list_new_i
            Q_list_new[i] = Q_list_new_i
            # Accumulate the *damped* delta so r stays consistent with r = r0 + sum(r_list)
            r_delta += r_list_new_i.reshape(-1, 1) - r_list[i].reshape(-1, 1)
            Q_delta += Q_list_new_i - Q_list[i]

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
        
        if np.linalg.norm(mu - mu_old) < args['ep_tol']:
            return r, Q, r_list, Q_list
        # elif np.linalg.norm(mu - mu_old) < max_gap:
        #     max_gap = np.linalg.norm(mu - mu_old)
        #     best_r = np.copy(r); best_Q = np.copy(Q); best_r_list = np.copy(r_list); best_Q_list = np.copy(Q_list)

    return r, Q, r_list, Q_list

def _device_site_update(i, r, Q, r_list, Q_list, diff_lk_list, model, args):
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

    # Always symmetrize (cheap, always safe): Stan's own symmetry check is much
    # stricter than check_symmetric's tolerance, so a matrix that looks "close enough"
    # here can still be rejected by multi_normal_lpdf on the Stan side.
    Sigma_cavity = (Sigma_cavity + Sigma_cavity.T) / 2

    dat = {
        "N": len(lk_list),
        "Delta": args['delta'],
        "sigma_0": args['sigma0'].item(),
        "Delta_l_k": lk_list,
        "k": k_list,
        "mu_i": mu_cavity.flatten(),
        "Sigma_i": Sigma_cavity,
        "mu_mu": args['mu_mu'].flatten().tolist(), "Sigma_mu": args['Sigma_mu'].tolist(),
        "shape_Sigma": args['shape_Sigma'].item(), "nu_Sigma": args['nu_Sigma'].tolist()
    }
    # treedepth = 10 if len(k_list) > 10 else 15
    for idx in range(20):
        with tempfile.TemporaryDirectory() as tmp_out, suppress_stdout_stderr():
            fit_pred = model.sample(data=dat, iter_sampling=args['num_samples'], iter_warmup=args['num_burnin'], chains=args['num_chains'], adapt_delta=0.999, inits=0.,
                                        show_progress=False, save_warmup=False, max_treedepth=12, output_dir=tmp_out)

            df_fit_pred = fit_pred.draws(inc_warmup=False)
            col_name = np.array(fit_pred.column_names)
            col_idx = [np.where(col_name=='beta[1]')[0][0], np.where(col_name=='beta[2]')[0][0]]
            beta_samples = df_fit_pred[:, :, col_idx]

        # Compute autorrelation
        rho_hat, R_hat, mix_chains_idx, flag = autocorr_estimate(beta_samples[args['num_burnin']:, :, :], args['max_lag'], args)

        if flag == 1:
            break

    tau = (1 + 2 * np.sum(rho_hat, axis=0)).astype(int)
    tau = np.where(tau > 0, tau, 1)

    beta_mean = np.zeros(2)
    beta_std = np.zeros(2)
    for j in range(2):
        samples_j = beta_samples[args['num_burnin']::tau[j], mix_chains_idx[j], j].ravel()
        beta_mean[j] = np.mean(samples_j)
        beta_std[j] = np.std(samples_j)

    return beta_mean, beta_std

def device_posterior_update(r, Q, r_list, Q_list, diff_lk_list, model, args):
    d = args['d']; N = args['N']
    mu_list = np.zeros((N, d))
    sigma_list = np.zeros((N, d))

    site_results = Parallel(n_jobs=args.get('site_n_jobs', 1))(
        delayed(_device_site_update)(i, r, Q, r_list, Q_list, diff_lk_list, model, args)
        for i in range(N)
    )
    for i, (beta_mean, beta_std) in enumerate(site_results):
        mu_list[i] = beta_mean
        sigma_list[i] = beta_std

    return mu_list, sigma_list, 0