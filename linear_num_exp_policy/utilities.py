import matplotlib.ticker as ticker
import matplotlib.pyplot as plt 
import pandas as pd
from pathlib import Path
import re

import os
import numpy as np

def MSE(y_pred, y_test):
    return np.mean(np.power(y_pred - y_test, 2), axis=0)

def MAPE(y_pred, y_test):
    return np.mean(np.abs((y_pred - y_test)), axis=0)

def load_data(args):
    dataset_name, dataset_num = extract_prefix_and_number(args['dataset'])
    thresholds = {} 
    
    if dataset_name == 'nasa':
        parent_path = Path.cwd().parent / Path("dataset") / Path(dataset_name)
        file_path = Path(f"train_FD00{int(dataset_num)}.txt")
        
        df = pd.read_csv(parent_path / file_path, sep=" ", header=None)
        df.drop(columns=[26, 27], inplace=True)
        
        columns = ['unit_number','time_in_cycles','setting_1','setting_2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
                   'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32']
        df.columns = columns
        
        # Isolate the target columns
        df = df[['unit_number', 'time_in_cycles', 'T24']].copy()
        
        # 1. Calculate robust baseline (mean of first 10 cycles for each unit)
        span_value = 15
        baselines = df[df['time_in_cycles'] <= 10].groupby('unit_number')['T24'].mean()
        df['T24_adj'] = df['T24'] - df['unit_number'].map(baselines)
        df['T24_smoothed'] = df.groupby('unit_number')['T24_adj'].transform(lambda x: x.ewm(span=span_value, adjust=False).mean())
        
        # 2. Calculate RUL
        max_cycles = df.groupby('unit_number')['time_in_cycles'].transform('max')
        df['RUL'] = max_cycles - df['time_in_cycles']
        
        # 3. Pivot and isolate overlapping end-of-life phase
        df_pivot = df.pivot(index='RUL', columns='unit_number', values='T24_adj')
        df_pivot = df_pivot.dropna().sort_index(ascending=False)
        
        # --- NEW: Cut off the signal at the degradation horizon 'K' ---
        if 'K' in args:
            # Keep only the final K cycles (e.g., if K=50, keep RUL 49 down to 0)
            df_pivot = df_pivot[df_pivot.index < args['K']]
        
        # 4. Force the start of this specific K-window to be exactly 0
        # By doing this AFTER the cut, we measure the delta strictly within the horizon
        df_pivot = df_pivot - df_pivot.iloc[0]
        
        # 5. Calculate new thresholds based on the isolated horizon curve
        failure_values = df_pivot.loc[0]
        thresholds['mean_failure'] = failure_values.mean()
        thresholds['conservative_failure'] = failure_values.quantile(0.2)
        
        # 6. Reset index and format columns
        df_pivot = df_pivot.reset_index(drop=True)
        df_pivot.columns = [f'unit_{int(col)}' for col in df_pivot.columns]

        # Filter only some sites
        if args.get('scenario') == 1:
            df_pivot = df_pivot.sample(axis=1, n=args['num_site'])

    else:
        print("Specify a valid argument for dataset")
        df_pivot = None

    return df_pivot, thresholds

def process_dataframe(mat, args):
    if args['scenario'] == 1:
        T = mat.shape[0]
        N = mat.shape[1] 

        # Initialize placeholder
        hist_lk_list = [[[]] for n in range(N)]
        diff_hist_lk_list = [[[]] for n in range(N)]

        m = 0 # Retain the same shape as in the simulated dataset
        # Loop over every element
        for t in range(1, T):
            for n in range(N):
                hist_lk_list[n][m].append((t, mat.iloc[t, n]))
                diff_hist_lk_list[n][m].append((t, mat.iloc[t, n] - mat.iloc[t-1, n]))

    elif args['scenario'] == 2:
        # Randomly shuffle the sites
        site_nums = np.arange(100)
        np.random.shuffle(site_nums)

        # Initialize placeholder
        hist_lk_list = [[[] for m in range(args['M'])] for n in range(args['N'])]
        diff_hist_lk_list = [[[] for m in range(args['M'])] for n in range(args['N'])]

        for n in range(args['N']):
            for m in range(args['M']):
                k = 1
                idx = site_nums[n*10 + m]

                while mat.iloc[k, idx] <= args['r_limit']:
                    hist_lk_list[n][m].append((k, mat.iloc[k, idx]))
                    diff_hist_lk_list[n][m].append((k, mat.iloc[k, idx] - mat.iloc[k-1, idx]))
                    k += 1
    else:
        print("Invalid scenario")

    return hist_lk_list, diff_hist_lk_list

def extract_prefix_and_number(input_string):
    """
    Extracts the prefix and number from a string using a regular expression.

    Parameters:
    input_string (str): The input string to process.

    Returns:
    tuple: A tuple containing the prefix and the number as strings, or (None, None) if not found.
    """
    # Use regular expression to capture both the prefix (letters) and the trailing number
    match = re.match(r'([a-zA-Z]+)(\d+)$', input_string)
    
    if match:
        prefix = match.group(1)  # Captures the alphabetic prefix
        number = match.group(2)  # Captures the numeric suffix
        return prefix, number
    
    return None, None

def plotaction(pi, args):
    fig, ax = plt.subplots()

    action = pi.T#pi[1:-1].T
    ax.imshow(action, origin="lower")

    def format_yticks(y, pos):
        return "%.2f"%(y * args["state_upper_limit"]/(args["total_grid"]-1))

    # Create a new formatter using the format_yticks function
    formatter = ticker.FuncFormatter(format_yticks)

    # Apply the formatter to the y-axis
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.savefig('actions.png',bbox_inches="tight")

    return

def calc_cost(k, args, cost_type):
    replace_cost = args['c1'] if cost_type == 'urgent' else args['c2']
    if args["gamma"] == 1:
        cost = k * args["c3"] + replace_cost
    else:
        cost = args["c3"]*(1-args["gamma"]**(k+1))/(1-args["gamma"]) + replace_cost*args["gamma"]**k    
    return cost

# from https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def CholeskyAlgorithm(H, beta):
    # Initialize eta
    diag_min = np.min(np.diag(H))
    if diag_min > 0:
        eta = 0
    else:
        eta = -diag_min + beta

    # Initialize flag
    L = None

    while L is None:
        try:
            L = np.linalg.cholesky(H + eta * np.eye(len(H)))
        except np.linalg.LinAlgError:
            eta = max(2*eta, beta)
    return H + eta * np.eye(len(H))
