import numpy as np
import scipy
import itertools
import copy
import os
import time
import pandas as pd
from contextlib import nullcontext

import multiprocessing as mp
from tqdm import tqdm
from joblib import Parallel, delayed
from cmdstanpy import CmdStanModel
import matplotlib.pyplot as plt
from scipy import stats

from utilities import suppress_stdout_stderr, check_symmetric, CholeskyAlgorithm

def reset_EP_priors(ev_initial, cov_initial, args):
    d = args['d']
    r0 = np.linalg.solve(cov_initial, ev_initial)
    Q0 = np.linalg.inv(cov_initial)
    
    # Initialize r_i and Q_i
    r_list = np.zeros((args['N'], d + d**2))
    Q_list = np.zeros((args['N'], d+d**2, d+d**2))

    r_i = np.linalg.solve(args['site_Sigma'], args['site_mu'])
    Q_i = np.linalg.inv(args['site_Sigma'])

    for n in range(args['N']):
        r_list[n] = r_i.flatten()
        Q_list[n] = Q_i

    r = r0 + np.sum(r_list, axis=0).reshape(-1, 1)
    Q = Q0 + np.sum(Q_list, axis=0)

    return r, Q, r_list, Q_list

def update_posterior(args, diff_lk_input, noise, rng):    
    # Initialize memories
    mu_hist = np.zeros((args['n_chains'], args['gibbs_T']))
    Sigma_hist = np.zeros((args['n_chains'], args['gibbs_T']))
    betas_hist = np.zeros((args['N'], args['n_chains'], args['gibbs_T']))

    # Run Gibbs
    for chain in range(args['n_chains']):
        # Randomly sample the starting point according to the prior
        mu = stats.norm.rvs(loc=args['mu_mu'], scale=args['Sigma_mu'], random_state=rng)
        Sigma = stats.invgamma.rvs(a=args['alpha_Sigma'], scale=args['beta_Sigma'], random_state=rng)
        betas = stats.norm.rvs(loc=mu, scale=Sigma, size=args['N'], random_state=rng)
        
        for t in range(args['gibbs_T']):      
            # Get a sample from mu
            precision = 1 / args['Sigma_mu']**2 + args['N'] / Sigma**2
            linear_shift = args['mu_mu'] / args['Sigma_mu']**2 + np.sum(betas) / Sigma**2

            mean = linear_shift / precision
            var = 1 / precision

            new_mu = stats.norm.rvs(loc=mean, scale=np.sqrt(var), random_state=rng)

            # Get a sample from Sigma
            alpha_param = args['alpha_Sigma'] + args['N'] / 2
            beta_param = args['beta_Sigma'] + np.sum(betas - new_mu)**2 / 2
            
            new_Sigma = np.sqrt(stats.invgamma.rvs(a=alpha_param, scale=beta_param, random_state=rng))

            # Get samples for betas
            # Initialize empty betas
            new_betas = np.zeros(args['N'])

            # Loop over every site
            for n in range(args['N']):
                lk_list_n = np.stack(diff_lk_input[n])
                k_list = lk_list_n[:, 0]
                diff_lk_list = lk_list_n[:, 1]

                num_data = len(diff_lk_list)
                state_info = np.sum(diff_lk_list) + rng.laplace(scale=args['lap_noise'])
                # print(noise[int(k_list[-1])])
                
                precision = num_data * args['delta']**2 / args['sigma0']**2 + 1 / Sigma**2
                linear_shift = state_info * args['delta'] / args['sigma0']**2 + mu / Sigma**2
                
                mean = linear_shift / precision
                var = 1 / precision
                new_betas[n] = stats.norm.rvs(loc=mean, scale=np.sqrt(var), random_state=rng)

            # Update params
            mu = new_mu; Sigma = new_Sigma; betas = new_betas
            
            # Append history
            mu_hist[chain, t] = mu
            Sigma_hist[chain, t] = Sigma
            betas_hist[:, chain, t] = betas
    
    # Take only the samples after the warm-up period
    mu_hist = mu_hist[:, args['warm_up']:]
    Sigma_hist = Sigma_hist[:, args['warm_up']:]
    betas_hist = betas_hist[:, :, args['warm_up']:]
    
    # Collapse the array
    mu_hist = mu_hist.flatten()
    Sigma_hist = Sigma_hist.flatten()
    betas_hist = betas_hist.reshape(args['N'], -1)
    
    mean_betas = np.mean(betas_hist, axis=-1)
    std_betas = np.std(betas_hist, axis=-1)
    
    return mean_betas, std_betas

def update_isolated_posterior(args, diff_lk_input):
    d = args['d']
    mu0_mean = np.zeros((args['N'], args['d']))
    mu0_std = np.zeros((args['N'], args['d']))
    
    for n in range(args['N']):
        lk_list_n = np.stack(diff_lk_input[n])
        num_data = len(lk_list_n)
        new_diff_lk = lk_list_n[:, 1].reshape(-1, 1)
        A = 1 / args['Sigma_mu']**2 + num_data * args['delta']**2 / (args['sigma0']**2)
        b = args['mu_mu'] / args['sigma0']**2 + np.sum(new_diff_lk)*args['delta'] / args['sigma0']**2

        mu0_mean[n] = b / A
        mu0_cov =  1 / A

        mu0_std[n] = np.sqrt(np.diag(mu0_cov))

    return mu0_mean, mu0_std

def update_EP_posterior(r, Q, r_list, Q_list, diff_lk_input, model, args, rng):
    d = args['d']
    ep_show_console = bool(args.get('ep_show_console', False))
    ep_log_site = bool(args.get('ep_log_site', False))
    for c in range(args['C']):
        # print(f"[EP] round={c+1}/{args['C']} start")
        # Device-side
        r_delta = 0
        Q_delta = 0

        r_list_new = np.zeros((args['N'], 2))
        Q_list_new = np.zeros((args['N'], 2, 2))

        for i in range(args['N']):
            # Stack data
            lk_list_n = np.stack(diff_lk_input[i])
            k_list = lk_list_n[:, 0]
            lk_list = lk_list_n[:, 1]
            # print(lk_list)

            # Subtract old r from r0 and old Q from Q0 to remove the impact of old params
            r_cavity = r - r_list[i].reshape(-1, 1)
            Q_cavity = Q - Q_list[i]
            
            # Run MCMC
            mu_cavity = np.linalg.solve(Q_cavity, r_cavity)
            Sigma_cavity = CholeskyAlgorithm(np.linalg.inv(Q_cavity), args['epsilon'])
            if ep_log_site:
                print(
                    f"[EP] round={c+1} site={i} "
                    f"mu_cavity={mu_cavity.flatten().tolist()} "
                    f"sigma_diag={np.diag(Sigma_cavity).tolist()} "
                    f"n_obs={len(lk_list)}"
                )

            # Check symmetry
            if not check_symmetric(Sigma_cavity):
                off_diag = max(Sigma_cavity[0, 1], Sigma_cavity[1, 0])
                Sigma_cavity[0, 1] = Sigma_cavity[1, 0] = off_diag

            if Sigma_cavity[0, 1] < 1e-4:
                Sigma_cavity[0, 1] = Sigma_cavity[1, 0] = 0

            # Use stan
            dat = {
                "N": len(lk_list), "Delta": args['delta'], "sigma_0": args['sigma0'].item(),
                "Delta_l_k": lk_list.tolist(), "k": k_list.tolist(), "mu_i": mu_cavity.flatten(), "Sigma_i": Sigma_cavity
            }
            stan_seed = int(rng.integers(1, 2**31 - 1))
            try:
                context_mgr = suppress_stdout_stderr() if not ep_show_console else nullcontext()
                with context_mgr:
                    fit = model.sample(
                        data=dat,
                        iter_sampling=2500,
                        iter_warmup=1000,
                        parallel_chains=4,
                        seed=stan_seed,
                        show_console=ep_show_console,
                        show_progress=False
                    )

            except RuntimeError as e:
                print(
                    f"[EP][ERROR] hybrid sampling failed "
                    f"(round={c+1}, site={i}, stan_seed={stan_seed}, n_obs={len(lk_list)})"
                )
                print(f"[EP][ERROR] mu_cavity={mu_cavity.flatten().tolist()}")
                print(f"[EP][ERROR] sigma_diag={np.diag(Sigma_cavity).tolist()}")
                print(f"[EP][ERROR] sigma_matrix=\n{Sigma_cavity}")
                raise RuntimeError(
                    f"EP hybrid sampling failed at round={c+1}, site={i}, stan_seed={stan_seed}. "
                    f"Set ep_show_console=True to see Stan console output."
                ) from e

            df_fit = fit.draws_pd()
            hybrid_samples = df_fit[['mu', 'tau']].values

            # Extract r_hybrid and Q_hybrid
            mu_hybrid = np.mean(hybrid_samples, axis=0).reshape(-1, 1)
            cov_hybrid = np.cov(hybrid_samples.T)
            # print(f"C {c} Site {i} | new mu (stan) = {mu_hybrid.flatten()}")
            # print(cov_hybrid)

            # # Calc stats
            # mu_hybrid = np.sum(weighted_samples, axis=0) / np.sum(weight)
            # mu_hybrid = mu_hybrid.reshape(-1, 1)
            # cov_hybrid = np.cov(ind_samples.T)
            # print(f"C {c} Site {i} | new mu = {mu_hybrid.flatten()}")
            # print(cov_hybrid)
            # print('-'*30)
            
            r_hybrid = np.linalg.solve(cov_hybrid, mu_hybrid)
            Q_hybrid = np.linalg.inv(cov_hybrid)

            # Update the contribution to the central prior
            r_delta += r_hybrid - r
            Q_delta += Q_hybrid - Q

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
        # print(f"[EP] round={c+1}/{args['C']} mu_gap={np.linalg.norm(mu - mu_old):.6e}")
        if np.linalg.norm(mu - mu_old) < args['ep_tol']:
            return r, Q, r_list, Q_list

    return r, Q, r_list, Q_list

def device_posterior_update(r, Q, r_list, Q_list, diff_lk_list, model, args, rng):
    mu_list = []
    sigma_list = []
    ep_show_console = bool(args.get('ep_show_console', False))
    
    d = args['d']
    N = args['N']

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
        # print(f"Site {i+1} | {mu_cavity.flatten()} | {Sigma_cavity.flatten()}")

        if not check_symmetric(Sigma_cavity):
            off_diag = max(Sigma_cavity[0, 1], Sigma_cavity[1, 0])
            Sigma_cavity[0, 1] = Sigma_cavity[1, 0] = off_diag

        if Sigma_cavity[0, 1] < 1e-8:
            Sigma_cavity[0, 1] = Sigma_cavity[1, 0] = 0


        dat = {
            "N": len(lk_list),
            "Delta": args['delta'],
            "sigma_0": args['sigma0'].item(),
            "Delta_l_k": lk_list.tolist(),
            "k": k_list.tolist(),
            "mu_i": mu_cavity.flatten().tolist(),
            "Sigma_i": Sigma_cavity
        }

        stan_seed = int(rng.integers(1, 2**31 - 1))
        context_mgr = suppress_stdout_stderr() if not ep_show_console else nullcontext()
        try:
            with context_mgr:
                fit_pred = model.sample(
                    data=dat,
                    iter_sampling=5000,
                    iter_warmup=2500,
                    parallel_chains=4,
                    adapt_delta=0.99,
                    seed=stan_seed,
                    show_console=ep_show_console,
                    show_progress=False
                )
        except RuntimeError as e:
            print(
                f"[EP][ERROR] predictive sampling failed "
                f"(site={i}, stan_seed={stan_seed}, n_obs={len(lk_list)})"
            )
            print(f"[EP][ERROR] mu_cavity={mu_cavity.flatten().tolist()}")
            print(f"[EP][ERROR] sigma_diag={np.diag(Sigma_cavity).tolist()}")
            raise RuntimeError(
                f"EP predictive sampling failed at site={i}, stan_seed={stan_seed}. "
                f"Set ep_show_console=True to see Stan console output."
            ) from e

        df_fit_pred = fit_pred.draws_pd()
        beta_samples = df_fit_pred['beta_i'].values

        beta_mean = np.mean(beta_samples)
        beta_std = np.std(beta_samples)

        mu_list.append(beta_mean)
        sigma_list.append(beta_std)

    mu_list = np.stack(mu_list)
    sigma_list = np.stack(sigma_list)

    return mu_list, sigma_list
