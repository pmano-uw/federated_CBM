import matplotlib.ticker as ticker
import matplotlib.pyplot as plt 

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

def within_chain_var(mat):
    N = mat.shape[0]
    M = mat.shape[1]

    return W, s_m_2

def between_chain_var(mat):
    N = mat.shape[0]
    M = mat.shape[1]
    return B

def autocorr_estimate(mat, max_lag, args):
    N = mat.shape[0]
    M = mat.shape[1]
    rho_hat_m = np.zeros((max_lag, M, 2))
    rho_hat = np.zeros((max_lag, 2))

    # Calculate autocorrelation
    for m in range(M):
        for i in range(2):
            with suppress_stdout_stderr():
                autocorr_i = acf(mat[:, m, i], nlags=max_lag, fft=True)[1:]
            rho_hat_m[:, m, i] = autocorr_i

    mix_chains_idx = []
    len_mix = []
    for i in range(2):
        # print(np.mean(np.abs(rho_hat_m[:,:,i]), axis=0))
        mix_chain_idx = np.argwhere(np.mean(np.abs(rho_hat_m[:,:,i]), axis=0) < args['rho_threshold']).ravel().tolist()
        if len(mix_chain_idx) <= 1:
            # print("No mixing chains")
            mix_chain_idx = np.array(range(M))
        mix_chains_idx.append(mix_chain_idx)
        len_mix.append(len(mix_chain_idx))
    # print(mix_chains_idx)
    W = np.zeros(2); B = np.zeros(2)
    s_m_2 = []
    for i in range(2):
        # Calc within chain variance
        mat_i = mat[:, mix_chains_idx[i], i]
        mean_m = np.mean(mat_i, axis=0)
        squared_diff_m = np.square(mat_i-mean_m)
        s_m_squared = np.sum(squared_diff_m, axis=0) / (N-1)
        W[i] = np.mean(s_m_squared)
        s_m_2.append(s_m_squared.ravel())

        # Calc between chain variance
        mean_dot_m = np.mean(mat[:, mix_chains_idx[i], i], axis=0)
        mean_dot_dot = np.mean(mean_dot_m, axis=0)
        B[i] = (N/(len_mix[i]-1)) * np.sum(np.power(mean_dot_m - mean_dot_dot, 2), axis=0)

    var_hat = ((N-1) / N) * W + (1/N) * B
    R_hat = np.sqrt(var_hat / W)

    for i in range(2):
        curr_rho_hat = rho_hat_m[:, mix_chains_idx[i], i]
        s_m = s_m_2[i]
        rho_hat_s = curr_rho_hat * s_m
        rho_hat[:, i] = 1 - (W[i] - np.mean(rho_hat_s, axis=1)) / var_hat[i]
    return rho_hat, R_hat, mix_chains_idx