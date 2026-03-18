import numpy as np
import copy
import itertools
import time

from scipy import stats

def value_iteration(mean, std, bins, args, threshold=0.0001):  
    converge = 0
    
    c1 = args['c1']
    c2 = args['c2']
    c3 = args['c3']

    gamma = args['gamma']
    
    value = np.zeros((args['K'], args['S']))
    pi = np.zeros((args['K'], args['S']))

    new_bins = copy.deepcopy(bins)
    new_bins[1:] = bins[1:] - np.diff(bins)[0] / 2
    new_bins = np.append(new_bins, args['s1_limit'])

    # Pre calculate transition probability to avoid repetition when doing value iteration 
    prob = calc_transition_prob(new_bins, bins, mean, std, args)

    for i in range(args['max_iter']):
        old_value = copy.deepcopy(value)
        
        # Set value on the edge
        value[args['K']-1] = np.where(new_bins <= args['r_limit'], c3, c1)
        pi[args['K']-1] = np.where(new_bins <= args['r_limit'], 0, 1)

        cost_A0 = copy.deepcopy(value[:-1, :]) # Cost of "not" replacing machines (matrix K x S)
        for k in range(args['K']-1):
            future_cost_k = c3 + gamma * value[k+1, :]
            cost_A0[k, :] = prob[k,:,:] @ future_cost_k.T
        cost_A1 = c2 + gamma * value[0, 0] # Cost of replacing machine (scalar)

        # Set value and pi
        value[:-1, :] = np.where(cost_A0 >= cost_A1, cost_A0, cost_A1)
        pi[:-1, :] = np.where(cost_A0 >= cost_A1, 0, 1)

        # Reset value and pi if lk exceeds threshold
        value = np.where(new_bins > args['r_limit'], c1 + gamma * value[0, 0], value)
        pi = np.where(new_bins > args['r_limit'], 1, pi)
        
        # if i % 5 == 0:
        #     print(f"Iter {i+1}: diff value = {np.linalg.norm(old_value - value)}")
        if np.linalg.norm(old_value - value) < threshold:
            converge = 1
            break 
    if converge == 0:
        print("Reach maximum iteration before convergent")

    return value, pi, prob

def VI_opt(pi_hat, prob, new_bins, args, threshold=0.0001):
    converge = 0
    
    c1 = args['c1']
    c2 = args['c2']
    c3 = args['c3']

    gamma = args['gamma']
    
    value = np.zeros((args['K'], args['S']))

    for i in range(args['max_iter']):
        old_value = copy.deepcopy(value)
        pis_trim = pi_hat[:-1, :]

        # Set value on the edge
        value[args['K']-1] = np.where(new_bins <= args['r_limit'], c3, c1)

        cost_A0 = copy.deepcopy(value[:-1, :]) # Cost of "not" replacing machines (matrix K x S)
        for k in range(args['K']-1):
            future_cost_k = c3 + gamma * value[k+1, :]
            cost_A0[k, :] = prob[k,:,:] @ future_cost_k.T
        cost_A1 = c2 + gamma * value[0, 0] # Cost of replacing machine (scalar)

        # Set value and pi
        value[:-1, :] = np.where(pis_trim == 0, cost_A0, cost_A1)

        # Reset value and pi if lk exceeds threshold
        value = np.where(new_bins > args['r_limit'], c1 + gamma * value[0, 0], value)

        # if i % 5 == 0:
        #     print(f"Iter {i+1}: diff value = {np.linalg.norm(old_value - value)}")
        if np.linalg.norm(old_value - value) < threshold:
            converge = 1
            break 
    if converge == 0:
        print("Reach maximum iteration before convergent")

    return value


def calc_one_step_transition_prob(mean, std, lk, bins, delta, sigma):
    # Calculate mean and std for l_{k+1}
    mean_lk1 = lk + mean * delta
    std_lk1 = np.sqrt(std**2 * delta**2 + sigma**2)

    # Derive cdf of max of each bin
    normalized_bins = (bins - mean_lk1) / std_lk1
    cdf_bins = stats.norm.cdf(normalized_bins)
    
    prob = np.diff(cdf_bins)
    prob = np.insert(prob, 0, cdf_bins[0]) # Add prob for lk <= 0
    prob = np.append(prob, 1 - cdf_bins[-1])
    return prob

def calc_transition_prob(state_partition, bins, prior_mean, prior_std, args):
    # Matrix of K x S x S
    prob = np.zeros((args['K'], args['S'], args['S']))
    for k in range(args['K']):
        for s in range(args['S']):
            tk = args['delta'] * k
            lk = state_partition[s]
            # Update posterior distribution by approximated lk if k > 0
            if k > 0:
                a = ((tk**2) / (k*args['sigma0']**2)) + (1/(prior_std**2))
                b = (((lk - args['l_0']) * tk) / (k*args['sigma0']**2)) + (prior_mean/(prior_std**2))

                mu_k = b/a
                sigma_k = np.sqrt(1/a)
            else:
                mu_k = prior_mean
                sigma_k = prior_std
            
            prob[k, s, :] = calc_one_step_transition_prob(mu_k, sigma_k, lk, bins, args['delta'], args['sigma0'])
    return prob

def calc_transition_prob_opt(state_partition, bins, true_beta, args):
    prob = np.zeros((args['K'], args['S'], args['S']))
    for k in range(args['K']):
        for s in range(args['S']):
            tk = args['delta'] * k
            lk = state_partition[s]
            prob[k, s, :] = calc_one_step_transition_prob(true_beta, 0, lk, bins, args['delta'], args['sigma0'])

    return prob