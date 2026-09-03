import pystan

def main():
    schools_code = """
        data {
            int<lower=1> N;              // Total number of data
            int<lower=1> d;              // Number of dimensions
            real<lower=1> Delta;         // Lag time
            real<lower=0> sigma_0;       // Noise in Brownian motion

            vector[N] Delta_l_k;         // Observations' diff
            vector[N] k;                 // Time
            
            vector[d] mu_mu              // Prior mean for mu
            matrix[d, d] Sigma_mu        // Prior covarinace for mu
            
            vector[d] mu_tau             // Prior mean for tau (covariance of central server)
            matrix[d, d] Sigma_tau       // Prior covariance for tau

            // EP Approximation parameters
            vector[2*d] mu_i;            // Mean of EP approx.
            matrix[2*d, 2*d] Sigma_i;    // Sigma of EP approx.
        }
        parameters {
            vector[d] mu;                // mean of the central server
            vector[d] tau;               // variance of the central server
        }
        transformed parameters {
            vector[2*d] phi;             // concat mu and tau
            phi = append_row(mu, tau);
        }
        model {
            vector[N] t_k;               // Time * lag time
            vector[N] mu_k;              // Mean of each observation
            vector[N] sigma_k;           // Variance of each observation

            t_k = Delta * k;     
            mu_k = dot_product(mu, t_k);
            sigma_k = pow(sigma_0, 2) + quad_form(tau, t_k);       

            target += multi_normal_lpdf(phi | mu_i, Sigma_i);
            for (n in 1:N)
                target += normal_lpdf(l_k[n] | mu_k, sigma_k);
        }
    """
    
    schools_dat = {'N': 5,
                'y': [5, 4, 3, -10, 0]}

    sm = pystan.StanModel(model_code=schools_code)
    fit = sm.sampling(data=schools_dat, iter=1000, chains=4)
    print(fit)

if __name__ == "__main__":
    main() 