import numpy as np
import copy
import itertools
import time

from scipy import stats

def value_iteration(prior_mean, prior_std, bins, raw_bins, args, options='estimate'):  
    converge = 0
    
    c1 = args['c1']; c2 = args['c2']; c3 = args['c3']

    gamma = args['gamma']
    sigma0 = args['sigma0']
    Sigma1 = args['Sigma1']
    l_0 = args['l_0']
    threshold = args['threshold'] if options == 'estimate' else 1e-12
    
    S_square = args['S']**2
    value = np.zeros((args['K'], S_square))
    pi = np.zeros((args['K'], S_square))
    
    # Generate middle of the bins as representation of state
    new_bins_2d = []
    raw_new_bins = []
    flag = []
    
    for k in range(args['K']):
        raw_bins_k = raw_bins[k]
        
        new_bins_state_1 = copy.deepcopy(raw_bins_k[0])
        new_bins_state_2 = copy.deepcopy(raw_bins_k[1])

        state_1_space = np.diff(raw_bins_k[0])[0] 
        state_2_space = np.diff(raw_bins_k[1])[0]

        new_bins_state_1[1:] = raw_bins_k[0][1:] - state_1_space/ 2
        new_bins_state_1 = np.append(new_bins_state_1, args['s1_limit'])

        new_bins_state_2[1:] = raw_bins_k[1][1:] - state_2_space / 2
        new_bins_state_2 = np.append(new_bins_state_2, args['s1_limit'] * (k+1))
        
        new_bins_2d_k = np.stack(list(itertools.product(new_bins_state_1, new_bins_state_2)))
        raw_new_bins_k = (new_bins_state_1, new_bins_state_2)
        
        # Append middle of the bins
        new_bins_2d.append(new_bins_2d_k)
        raw_new_bins.append(raw_new_bins_k)
        
        # Flag some spaces to white
        flag_k = new_bins_2d_k[:, 1] <= new_bins_2d_k[:, 0] * (k+1)
        flag.append(flag_k)
    
    new_bins_2d = np.stack(new_bins_2d)
    raw_new_bins = np.stack(raw_new_bins)
    
    trans_prob_start_time = time.time()
    
    # Pre calculate transition probability to avoid repetition when doing value iteration 
    if options == 'estimate':
        prob = calc_transition_prob(new_bins_2d, raw_new_bins, bins, prior_mean, prior_std, args)
    elif options == 'optimal':
        prob = calc_optimal_transition_prob(new_bins_2d, raw_new_bins, bins, prior_mean, args)
    prob = np.abs(prob)
    
    trans_prob_end_time = time.time()
    
    # Time how long the VI runs
    start = time.time()
   
    lk_matrix = copy.deepcopy(new_bins_2d[:, :, 0])
    
    for i in range(args['max_iter']):
        old_value = copy.deepcopy(value)
        
        # Set value on the edge
        value[args['K']-1] = np.where(lk_matrix[args['K']-1] <= args['r_limit'], c3, c1)
        pi[args['K']-1] = np.where(lk_matrix[args['K']-1] <= args['r_limit'], 0, 1)

        cost_A0 = copy.deepcopy(value[:-1, :]) # Cost of "not" replacing machines (matrix K x S)
        for k in range(args['K']-1):
            future_cost_k = c3 + gamma * value[k+1, :]
            cost_A0[k, :] = prob[k,:,:] @ future_cost_k.T
        cost_A1 = c2 + gamma * value[0, 0] # Cost of replacing machine (scalar)

        # Set value and pi
        value[:-1, :] = np.where(cost_A0 >= cost_A1, cost_A0, cost_A1)
        pi[:-1, :] = np.where(cost_A0 >= cost_A1, 0, 1)

        # Reset value and pi if lk exceeds threshold
        value = np.where(lk_matrix > args['r_limit'], c1 + gamma * value[0, 0], value)
        pi = np.where(lk_matrix > args['r_limit'], 1, pi)
            
#         if i % 200 == 0:
#         print(f"Iter {i+1}: gap = {np.linalg.norm(old_value - value)}")
        if np.linalg.norm(old_value - value) < threshold:
            converge = 1
            break 
    if converge == 0:
        print("Reach maximum iteration before convergent")
        
    end = time.time()
    elapsed = end - start
    elapsed_prob = trans_prob_end_time - trans_prob_start_time

    # print(f"VI took {elapsed // 60:.4f} mins {elapsed % 60:.4f} seconds | calc. prob took {elapsed_prob // 60:.4f} mins {elapsed_prob % 60:.4f} seconds")
    
    # Prevent signed zero
    pi = np.abs(pi); value = np.abs(value)
    return value, pi, prob, flag

def calc_one_step_transition_prob(s_k_index, mean, std, s_k, v_array, h_array, new_bins, s2_limit, args):
    # Declare variable
#     print(f"At state {s_k_index} ({s_k})")
#     print(f"mean = {mean} | std = {std}")

    v_counter = s_k_index[0]; h_counter = s_k_index[1];
    ss_1, ss_2 = new_bins
    prob = np.zeros((args['S'], args['S']))
    curr_ss1 = s_k[0]; curr_ss2 = s_k[1]
    intersection_list = [s_k]
    
    while np.abs(curr_ss1 - args['s1_limit']) > args['epsilon'] and np.abs(curr_ss2 - s2_limit) > args['epsilon']:
        next_v_xaxis = v_array[v_counter+1, 0]
        next_h_xaxis = h_array[h_counter+1, 0]
        if np.abs(next_h_xaxis - next_v_xaxis) <= args['epsilon']:
            end_point = v_array[v_counter+1, :]
            starting_point = intersection_list[-1]
            
            new_v_counter = v_counter + 1
            new_h_counter = h_counter + 1
            
        elif next_v_xaxis > next_h_xaxis:
            end_point = h_array[h_counter+1, :]
            starting_point = intersection_list[-1]
            
            new_v_counter = v_counter
            new_h_counter = h_counter + 1
            
        elif next_v_xaxis < next_h_xaxis:
            end_point = v_array[v_counter+1, :]
            starting_point = intersection_list[-1]
            
            new_v_counter = v_counter + 1
            new_h_counter = h_counter

        # Update posterior for current state at (v_counter, h_counter)
        # Transform end_point and starting_point into x_1 and x_2
        x_1 = (starting_point[0] - s_k[0]) / args['delta']
        x_2 = (end_point[0] - s_k[0]) / args['delta']
        
        # Normalize x_1 and x_2
        z_1 = (x_1 - mean) / std
        z_2 = (x_2 - mean) / std
        
        # Find standard normal CDF of x_1 and x_2
        F_1 = stats.norm.cdf(z_1)
        F_2 = stats.norm.cdf(z_2)

        transition_prob = F_2 if (starting_point == s_k).all() else F_2 - F_1
        prob[v_counter, h_counter] = transition_prob
#         print(f"transition to state {(v_counter, h_counter)} | transition prob. = {transition_prob}")
        
        # Update the list of interesecting points
        intersection_list.append(end_point)
        curr_ss1 = end_point[0]; curr_ss2 = end_point[1]
        
        # Update v_counter and h_counter
        v_counter_old = v_counter; v_counter = new_v_counter;
        h_counter_old = h_counter; h_counter = new_h_counter;

    intersection_list = np.stack(intersection_list)
    prob[v_counter, h_counter] = 1 - np.sum(prob)
#     print(f"transition to state {(v_counter, h_counter)} | transition prob. = {prob[v_counter, h_counter]}")
#     print('-------------------------')
    
    # Reshape prob
    prob = prob.flatten()
    
    return prob

def calc_transition_prob(state_partition, new_bins, bins, prior_mean, prior_cov, args):
    # Declare variables
    sigma0 = args['sigma0']
    extended_state_space = args['S'] ** 2
    
    # Inverse covaraince
    inv_prior_cov = np.linalg.inv(prior_cov)

    # Matrix of K x S x S
    prob = np.zeros((args['K'], extended_state_space, extended_state_space))
    for k in range(args['K']-1):
#         print(f"----------k = {k}-------------")
        ss_1, ss_2 = new_bins[k]
    
        delta_poly = (args['delta'] ** np.arange(1, 3)).reshape(-1, 1)
        tk_plus1 =  np.multiply(delta_poly, np.array([[1], [k+1]]))
    
        # Compute A_k in advance
        tk_list = np.vstack((np.repeat(1, k), np.arange(1, k+1)))
        T_list = np.multiply(delta_poly, tk_list)
        
        A_k = np.linalg.inv((1/sigma0**2) * (T_list @ T_list.T) + inv_prior_cov)
        t_A_k = tk_plus1.T @ A_k
        t_A_t_k = (t_A_k @ tk_plus1).item()
        
        # Compute sigma_k
        sigma_k = np.sqrt(sigma0**2 + t_A_t_k)
        
        for s in range(extended_state_space):
            # Retrieve state index
            state_index = (s // args['S'], s % args['S'])
            s_k = state_partition[k][s]
            
            # Calculate the state limit
            upper_limit_slope = args['delta'] * (k + 1.5)
            lower_limit_slope = args['delta'] / 2
            state_cond_upper = upper_limit_slope * s_k[0]
            state_cond_lower = lower_limit_slope * s_k[0]
            
            # Calculate t_k and t_{k+1}
            if s_k[1] <= state_cond_upper and s_k[1] >= state_cond_lower:
                if k > 0:                
                    # Calculate A_k and b_k which are components in our posterior
                    b_k = (1/sigma0**2) * s_k.reshape(-1, 1) + inv_prior_cov @ prior_mean
                else:
                    b_k = np.linalg.solve(prior_cov, prior_mean)

                # Posterior mean 
                mu_k = (t_A_k @ b_k).item()

                # p2 is the mean of predictive distribution
                p2 = (s_k[0] + mu_k * tk_plus1[0, 0], s_k[1] + mu_k * tk_plus1[1, 0])

                # Calculate slope
                m = tk_plus1[1, 0] / tk_plus1[0, 0]
                b = p2[1] - m * p2[0]

                # Calculate the intersection with the grid
                v_intersection = m * ss_1 + b
                h_intersection = (1/m) * (ss_2 - b)
                v_array = np.vstack((ss_1, v_intersection)).T
                h_array = np.vstack((h_intersection, ss_2)).T

                # Calculate transition prob.
                s2_limit = args['s1_limit'] * (k+1)
                prob[k, s, :] = calc_one_step_transition_prob(state_index, mu_k, sigma_k, s_k, v_array, h_array, new_bins[k], s2_limit, args)
    return prob

def calc_optimal_transition_prob(state_partition, new_bins, bins, true_beta, args):
    # Declare variables
    sigma0 = args['sigma0']
    extended_state_space = args['S'] ** 2

    # Matrix of K x S x S
    prob = np.zeros((args['K'], extended_state_space, extended_state_space))
    for k in range(args['K']-1):
#         print(f"----------k = {k}-------------")
        ss_1, ss_2 = new_bins[k]
    
        delta_poly = (args['delta'] ** np.arange(1, 3)).reshape(-1, 1)
        tk_plus1 =  np.multiply(delta_poly, np.array([[1], [k+1]]))
    
        # Compute A_k in advance
        tk_list = np.vstack((np.repeat(1, k), np.arange(1, k+1)))
        T_list = np.multiply(delta_poly, tk_list)
        
        # Compute sigma_k
        sigma_k = sigma0
        
        for s in range(extended_state_space):
            # Retrieve state index
            state_index = (s // args['S'], s % args['S'])
            s_k = state_partition[k][s]
            
            # Calculate the state limit
            upper_limit_slope = args['delta'] * (k + 1.5)
            lower_limit_slope = args['delta'] / 2
            state_cond_upper = upper_limit_slope * s_k[0]
            state_cond_lower = lower_limit_slope * s_k[0]
            
            # Calculate t_k and t_{k+1}
            if s_k[1] <= state_cond_upper and s_k[1] >= state_cond_lower:
                # Posterior mean 
                mu_k = np.dot(tk_plus1.T, true_beta).item()

                # p2 is the mean of predictive distribution
                p2 = (s_k[0] + mu_k * tk_plus1[0, 0], s_k[1] + mu_k * tk_plus1[1, 0])

                # Calculate slope
                m = tk_plus1[1, 0] / tk_plus1[0, 0]
                b = p2[1] - m * p2[0]

                # Calculate the intersection with the grid
                v_intersection = m * ss_1 + b
                h_intersection = (1/m) * (ss_2 - b)
                v_array = np.vstack((ss_1, v_intersection)).T
                h_array = np.vstack((h_intersection, ss_2)).T

                # Calculate transition prob.
                s2_limit = args['s1_limit'] * (k+1)
                prob[k, s, :] = calc_one_step_transition_prob(state_index, mu_k, sigma_k, s_k, v_array, h_array, new_bins[k], s2_limit, args)
    return prob