import itertools
import time
import copy
import random
# import pystan
import scipy
from scipy import stats
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from cmdstanpy import CmdStanModel

from posterior import update_posterior, update_isolated_posterior, update_EP_posterior, device_posterior_update, reset_EP_priors
from utilities import suppress_stdout_stderr, load_data, process_dataframe, get_pred, MSE, MAPE

def run_sim_linear(args):
    d = args['d']

    # Import data and scale by 100
    df = load_data(args)

    T = df.shape[0]; N = df.shape[1] 
    args['T'] = T; args['N'] = N
    hist_lk, hist_diff_lk = process_dataframe(df, args)

    # Calc index to create training/test data
    train_idx = np.array([int(0.25*T), int(0.5*T), int(0.75*T)])

    # Intialize empty list
    mse_gibb_hist = []; mape_gibb_hist = []
    mse_gibb_lap_1_hist = []; mape_gibb_lap_1_hist = []
    mse_gibb_lap_2_hist = []; mape_gibb_lap_2_hist = []
    mse_ep_hist = []; mape_ep_hist = []
    mse_iso_hist = []; mape_iso_hist = []
    
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
    with suppress_stdout_stderr():
        hybrid_dist_model = CmdStanModel(stan_file="stan_file/hybrid_posterior.stan", model_name="hybrid_distribution")
        predictive_model = CmdStanModel(stan_file="stan_file/predictive_posterior.stan", model_name="predictive_posterior")

    # Specify noise
    small_noise = 1
    large_noise = 3

    # Run sim
    for t in train_idx:
        hist_diff_lk_flat = []
        hist_diff_lk_lap_1 = []
        hist_diff_lk_lap_2 = []

        # Reset EP priors
        r, Q, r_list, Q_list = reset_EP_priors(ev_initial, cov_initial, args)

        for n in range(N):
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
        r, Q, r_list_new, Q_list_new = update_EP_posterior(r, Q, r_list, Q_list, hist_diff_lk_flat, hybrid_dist_model, args)
        time5 = time.time()
        beta_mean_ep, beta_std_ep = device_posterior_update(r, Q, r_list_new, Q_list_new, hist_diff_lk_flat, predictive_model, args)
        time6 = time.time()

        # print(f"Centralize time = {time2-time1:.4f} | Centralize noise time = {time3-time2:.4f} | Iso time = {time4-time3:.4f} | EP inner loop time = {time5-time4:.4f} | EP marginalize = {time6-time5:.4f}")

        # Create a test set
        y_test = df.iloc[t:, :].to_numpy()
        y_pred_start = df.loc[t-1, :].to_numpy()

        # Create estimates
        y_pred_gibb = get_pred(y_pred_start, beta_mean, T-t+1, args)
        y_pred_gibb_lap_1 = get_pred(y_pred_start, beta_mean_lap_1, T-t+1, args)
        y_pred_gibb_lap_2 = get_pred(y_pred_start, beta_mean_lap_2, T-t+1, args)
        y_pred_iso = get_pred(y_pred_start, beta_mean_iso.ravel(), T-t+1, args)
        y_pred_ep = get_pred(y_pred_start, beta_mean_ep, T-t+1, args)

        # Get MSE
        # print(beta_mean_iso.ravel())
        # print(y_pred_iso[:10, :5])
        # print(y_test[:10, :5])
        mse_gibb = MSE(y_pred_gibb, y_test); mape_gibb = MAPE(y_pred_gibb, y_test)
        mse_gibb_lap_1 = MSE(y_pred_gibb_lap_1, y_test); mape_gibb_lap_1 = MAPE(y_pred_gibb_lap_1, y_test)
        mse_gibb_lap_2 = MSE(y_pred_gibb_lap_2, y_test); mape_gibb_lap_2 = MAPE(y_pred_gibb_lap_2, y_test)
        mse_iso = MSE(y_pred_iso, y_test); mape_iso = MAPE(y_pred_iso, y_test)
        mse_ep = MSE(y_pred_ep, y_test); mape_ep = MAPE(y_pred_ep, y_test)

        print(f"time {t}, MSE | Gibb = {np.mean(mse_gibb):.4f} | Iso = {np.mean(mse_iso):.4f} | EP = {np.mean(mse_ep):.4f} | Lap 1 gap = {np.mean(mse_gibb_lap_1):.4f} | Lap 2 gap = {np.mean(mse_gibb_lap_2):.4f}")
        
        # Save mse and mape
        mse_gibb_hist.append(mse_gibb.tolist()); mape_gibb_hist.append(mape_gibb.tolist())
        mse_gibb_lap_1_hist.append(mse_gibb_lap_1.tolist()); mape_gibb_lap_1_hist.append(mape_gibb_lap_1.tolist())
        mse_gibb_lap_2_hist.append(mse_gibb_lap_2.tolist()); mape_gibb_lap_2_hist.append(mape_gibb_lap_2.tolist())
        mse_iso_hist.append(mse_iso.tolist());  mape_iso_hist.append(mape_iso.tolist())
        mse_ep_hist.append(mse_ep.tolist()); mape_ep_hist.append(mape_ep.tolist())

    return mse_gibb_hist, mse_gibb_lap_1_hist, mse_gibb_lap_2_hist, mse_iso_hist, mse_ep_hist, \
            mape_gibb_hist, mape_gibb_lap_1_hist, mape_gibb_lap_2_hist, mape_iso_hist, mape_ep_hist