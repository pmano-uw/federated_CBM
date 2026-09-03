import numpy as np
import scipy
from scipy import stats
import itertools
import copy

def get_prior(args, experiment):
    Sigma1_inv, Sigma2_inv, Sigma3_inv = np.linalg.inv(args['Sigma1']), np.linalg.inv(args['Sigma2']), np.linalg.inv(args['Sigma3'])
    
    # Initialize variables
    N = args['N']
    d = args['d']
    
    # Construct linear and quadratic matrix/vector
    if experiment == 'collaborative':
        B0 = np.zeros(((N+1)*d, 1))
        B0[:d] = Sigma3_inv @ args['mu3']

        A0 = np.zeros(((N+1)*d, (N+1)*d))
        A0[:d, :d] = N * Sigma2_inv + Sigma3_inv

        for i in range(args['N']):
            A0[:d, (i+1)*d:(i+2)*d] = - Sigma2_inv
            A0[(i+1)*d:(i+2)*d, :d] = - Sigma2_inv
            A0[(i+1)*d:(i+2)*d, (i+1)*d:(i+2)*d] = Sigma2_inv

        mu0_mean = np.linalg.solve(A0, B0)     
        mu0_cov = np.linalg.inv(A0)
        
    elif experiment == 'isolated' or experiment == 'EP':
        mu0_mean = np.kron(np.ones((N+1, 1)), args['mu3'])
        mu0_cov = np.kron(np.eye(N+1) ,args['Sigma3']+args['Sigma2'])

    return mu0_mean, mu0_cov

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

def get_betas_sample(args, mu, Sigma, betas, diff_lk_array, noise):
    # Initialize empty betas
    d = args['d']
    new_betas = np.zeros((args['N'], args['d']))
    
    for n in range(args['N']):
        Sigma_inv = np.linalg.inv(Sigma)
        
        lk_list_n = np.stack(diff_lk_array[n])
        k_list = lk_list_n[:, 0]
        tt_T = np.zeros((len(diff_lk_array[n]), d))
        for idx in range(d):
            tt_T[:, idx] = (lk_list_n[:, 0] ** idx) * (args['delta']**(idx+1))
    
        new_diff_lk = lk_list_n[:, 1].reshape(-1, 1)
        state_info = np.sum(np.multiply(new_diff_lk, tt_T), axis=0) + noise[int(k_list[-1])]

        A = Sigma_inv + tt_T.T @ tt_T / (args['sigma0']**2)
        b = (np.dot(Sigma_inv, mu) + state_info / (args['sigma0']**2)).reshape(-1, 1)
        mu0_mean = np.linalg.solve(A, b).flatten()
        mu0_cov = np.linalg.inv(A)
        new_betas[n] = stats.multivariate_normal.rvs(mean=mu0_mean.flatten(), cov=mu0_cov)

    return new_betas

def update_posterior(args, diff_lk_input, noise):
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
            new_betas = get_betas_sample(args, new_mu, new_Sigma, betas, diff_lk_input, noise)

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
    Sigma_hist = Sigma_hist.reshape(-1, args['d'])
    betas_hist = betas_hist.reshape(args['N'], -1, args['d'])
    
    mean_betas = np.mean(betas_hist, axis=1)
    cov_betas = np.zeros((args['N'], args['d'], args['d']))
    for site in range(args['N']):
        cov_betas[site, :, :] = np.cov(betas_hist[site].T)

    return mean_betas, cov_betas

def update_isolated_posterior(args, diff_lk_input):
    d = args['d']
    mu0_mean = np.zeros((args['N'], args['d']))
    mu0_cov = np.zeros((args['N'], args['d'], args['d']))
    
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
        mu0_cov[n] =  np.linalg.inv(A)

    mu0_cov = np.stack(mu0_cov)
    return mu0_mean, mu0_cov

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
            
            if ~check_symmetric(Sigma_cavity):
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

        if ~check_symmetric(Sigma_cavity):
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
        with suppress_stdout_stderr():
            fit_pred = model.sample(data=dat, iter_sampling=args['num_samples'], iter_warmup=args['num_burnin'], chains=args['num_chains'], adapt_delta=0.999, inits=1,
                                        show_progress=False, save_warmup=False)
    
        df_fit_pred = fit_pred.draws(inc_warmup=False)
        col_name = fit_pred.draws_pd(inc_warmup=False).columns
        col_idx = [np.where(col_name=='beta[1]')[0][0], np.where(col_name=='beta[2]')[0][0]]
        warmup_idx = args['num_burnin']
        beta_samples = df_fit_pred[:, :, col_idx]
        # print(beta_sademples)
        
        # Calculate sample variance of the chains
        N = beta_samples.shape[0]
        M = beta_samples.shape[1]

        # Compute autorrelation
        rho_hat, R_hat, mix_chains_idx = autocorr_estimate(beta_samples[warmup_idx:, :, :], args['max_lag'] ,args)
        tau = (1 + 2* np.sum(rho_hat, axis=0)).astype(int)
        tau = np.where(tau>0, tau, 1)

        beta_mean = np.zeros(2)
        beta_std = np.zeros(2)
        for j in range(2):
            samples_j = beta_samples[warmup_idx::tau[j], mix_chains_idx[j], j].ravel()
            beta_mean[j] = np.mean(samples_j)
            beta_std[j] = np.std(samples_j)

        # print(f"tau = {tau} | beta = {beta_mean} | R_hat = {R_hat} | ESS = {np.floor(N / tau)}")

        # Plot chains
        fig, ax = plt.subplots(2, 2)
        for j in range(2):
            ax[j, 0].plot(beta_samples[:, mix_chains_idx[j], j])
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

    return mu_list, sigma_list, 0


# def update_posterior(diff_lk_list, args, noise_sigma=None):
#     '''
#     diff_lk_list (list) - list of transition at timestep k minus transition at timestep k-1
#     args (dict) - dict containing parameters
#     t (int) - timestep at which the posterior is updated.
#     '''
#     d = args['d']
#     N = args['N']
    
#     ##### Initialize A #####
#     # Inverse of Sigma1, Sigma2, and Sigma 3 are going to be used a lot
#     Sigma1_inv, Sigma2_inv, Sigma3_inv = np.linalg.inv(args['Sigma1']), np.linalg.inv(args['Sigma2']), np.linalg.inv(args['Sigma3'])
    
#     mu1_block_A = np.zeros((N*d, N*d))
#     mu1_block_B = np.zeros((N*d, 1))
    
#     sum_m = 0
#     for n in range(N):
#         mu1_block_A[n*d:(n+1)*d, n*d:(n+1)*d] = Sigma2_inv
#         diff_lk_n = list(itertools.chain.from_iterable(diff_lk_list[n]))
#         for k, diff_lk in diff_lk_n:
#             tt_k = np.multiply(args['delta'] ** np.arange(1, d+1), k ** np.arange(d)).reshape(-1, 1)
#             tt_k_matrix = tt_k @ tt_k.T
#             denom = (args['sigma0']**2 + tt_k.T @ args['Sigma1'] @ tt_k).item()

#             mu1_block_A[n*d: (n+1)*d, n*d:(n+1)*d] += tt_k_matrix / denom
#             mu1_block_B[n*d: (n+1)*d] += (diff_lk * tt_k) / denom
        
#         # Add noise
#         laplacian_noise = 0 if noise_sigma is None else np.random.laplace(scale=noise_sigma)
#         mu1_block_B[n*d: (n+1)*d] += laplacian_noise

#     # Construct all blocks    
#     mu1_mu2_block = np.kron(np.ones((N, 1)), -Sigma2_inv)
#     mu2_block_A = N * Sigma2_inv + Sigma3_inv
#     mu2_block_B = Sigma3_inv @ args['mu3']
     
#     # Construct matrix A
#     quadratic_matrix = np.block([[mu2_block_A,      mu1_mu2_block.T],
#                                 [mu1_mu2_block,     mu1_block_A]])
#     linear_matrix = np.block([[mu2_block_B],
#                               [mu1_block_B]])
    
# #     # Solve for posterior
#     post_mean = np.linalg.solve(quadratic_matrix, linear_matrix)
#     post_cov = np.linalg.inv(quadratic_matrix)
    
#     return post_mean, post_cov

# def update_isolated_posterior(diff_lk_list, args):
#     d = args['d']
#     N = args['N']
#     mu0_mean = np.zeros(((N+1)*d, 1))
#     mu0_cov = np.zeros(((N+1)*d, (N+1)*d))
    
#     for n in range(args['N']):
#         diff_lk_n = list(itertools.chain.from_iterable(diff_lk_list[n]))
#         if args['Sigma1'].ndim == 1:
#             A = 1 / (args['Sigma2'] + args['Sigma3'])
#             B = args['mu3'] / (args['Sigma2'] + args['Sigma3'])
            
#             A = A.reshape(-1, 1); B = B.reshape(-1, 1)
#         else:     
#             A = np.linalg.inv(args['Sigma2'] + args['Sigma3'])
#             B = np.linalg.inv(args['Sigma2'] + args['Sigma3']) @ args['mu3']
        
#         for k, diff_lk in diff_lk_n:
#             tt_k = np.multiply(args['delta'] ** np.arange(1, d+1), k ** np.arange(d)).reshape(-1, 1)
#             tt_k_matrix = tt_k @ tt_k.T
#             denom = (args['sigma0']**2 + tt_k.T @ args['Sigma1'] @ tt_k).item()   
            
#             A += tt_k_matrix / denom
#             B += diff_lk * tt_k / denom

#         mu0_mean[(n+1)*d:(n+2)*d] = np.linalg.solve(A, B)
#         mu0_cov[(n+1)*d:(n+2)*d, (n+1)*d:(n+2)*d] = np.linalg.inv(A)

#     return mu0_mean, mu0_cov

# def update_EP_posterior(r_list, Q_list, r0, Q0, diff_lk_list, args):
#     N = args['N']
#     d = args['d']
#     # Device-side
#     r_delta = 0
#     Q_delta = 0

#     r_list_new = np.zeros((N, d))
#     Q_list_new = np.zeros((N, d, d))
#     for i in range(args['N']):
#         # Subtract old r from r0 and old Q from Q0 to remove the impact of old params
#         r_cavity = copy.deepcopy(r0 - r_list[i].reshape(-1, 1)) 
#         Q_cavity = copy.deepcopy(Q0 - Q_list[i])
        
#         # Initialize linear shift and precision mat of tilted dist.
#         r_hybrid = copy.deepcopy(r_cavity); Q_hybrid = copy.deepcopy(Q_cavity)
        
#         diff_lk_n = list(itertools.chain.from_iterable(diff_lk_list[i]))
#         # Compute f_k(mu_2)
#         for k, diff_lk in diff_lk_n:
#             tt_k = np.multiply(args['delta'] ** np.arange(1, d+1), k ** np.arange(d)).reshape(-1, 1)
#             tt_k_matrix = tt_k @ tt_k.T
#             denom = (args['sigma0']**2 + tt_k.T @ (args['Sigma1'] + args['Sigma2']) @ tt_k).item()    

#             r_hybrid += diff_lk * tt_k / denom
#             Q_hybrid += tt_k_matrix / denom

#         # Update the contribution to the central prior
#         r_delta += r_hybrid - r0
#         Q_delta += Q_hybrid - Q0
        
#         # Append new r_list and Q_list (local approximation)
#         r_list_new[i, :] = (r_hybrid - r_cavity).flatten()
#         Q_list_new[i, :, :] = Q_hybrid - Q_cavity

#     # Update global approximation
#     r0_new = r0 + r_delta
#     Q0_new = Q0 + Q_delta

#     return r0_new, Q0_new, r_list_new, Q_list_new

# def device_posterior_update(r0, Q0, r_list, Q_list, diff_lk_list, args):
#     d = args['d']
#     N = args['N']
#     mu0_mean = np.zeros(((N+1)*d, 1))
#     mu0_cov = np.zeros(((N+1)*d, (N+1)*d))

#     Sigma1_inv, Sigma2_inv, Sigma3_inv = np.linalg.inv(args['Sigma1']), np.linalg.inv(args['Sigma2']), np.linalg.inv(args['Sigma3'])

#     for i in range(args['N']):
#         # (Implicitly) calc mean and variance of q_sub_i(mu_2)
#         r_sub_i = copy.deepcopy(r0 - r_list[i].reshape(-1, 1))
#         Q_sub_i = copy.deepcopy(Q0 - Q_list[i])
#         Q_sub_i_inv = np.linalg.inv(Q_sub_i)
        
#         mu1_i_cov_inv = np.linalg.inv(Q_sub_i_inv + args['Sigma2'])
        
#         A = copy.deepcopy(mu1_i_cov_inv)
#         B = mu1_i_cov_inv.T @ Q_sub_i_inv @ r_sub_i
        
#         diff_lk_n = list(itertools.chain.from_iterable(diff_lk_list[i]))
        
#         for k, diff_lk in diff_lk_n:
#             tt_k = np.multiply(args['delta'] ** np.arange(1, d+1), k ** np.arange(d)).reshape(-1, 1)
#             tt_k_matrix = tt_k @ tt_k.T
#             denom = (args['sigma0']**2 + tt_k.T @ args['Sigma1'] @ tt_k).item()   

#             A += tt_k_matrix / denom
#             B += diff_lk * tt_k / denom
            
#         mu0_cov[(i+1)*d:(i+2)*d, (i+1)*d:(i+2)*d] = np.linalg.inv(A)
#         mu0_mean[(i+1)*d:(i+2)*d] = np.linalg.solve(A, B)

#     return mu0_mean, mu0_cov