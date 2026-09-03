data {
    int<lower=1> N;              // Total number of data
    real<lower=1> Delta;         // Lag time
    real<lower=0> sigma_0;       // Noise in Brownian motion

    vector[N] Delta_l_k;         // Observations' diff
    vector[N] k;                 // Time

    // EP Approximation parameters
    vector[5] mu_i;            // Mean of EP approx. over phi = [mu, vech(Sigma)]
    matrix[5, 5] Sigma_i;      // Sigma of EP approx.

    // Population-level priors, matching the centralized model exactly
    vector[2] mu_mu;
    matrix[2, 2] Sigma_mu;
    real<lower=0> shape_Sigma;
    matrix[2, 2] nu_Sigma;
}
parameters {
    cov_matrix[2] Sigma;      // covariance of the central server
    vector[2] beta;     // estimate of beta_i
    vector[2] mu;
}
transformed parameters {
    vector[2]           t_k;             // placeholder for time vector
    vector[N]           beta_scaled;
    vector[5]           phi;

    phi[1:2] = mu;
    phi[3] = Sigma[1, 1];
    phi[4] = Sigma[1, 2];
    phi[5] = Sigma[2, 2];

    for (n in 1:N){
        t_k = [Delta, k[n] * pow(Delta, 2)]';
        beta_scaled[n] = beta' * t_k;
    }
}
model {
    // Priors
    mu ~ multi_normal(mu_mu, Sigma_mu);
    Sigma ~ inv_wishart(shape_Sigma, nu_Sigma);

    // Sampling model
    target += multi_normal_lpdf(phi | mu_i, Sigma_i);
    target += multi_normal_lpdf(beta | phi[1:2], Sigma);
    target += normal_lpdf(Delta_l_k | beta_scaled, sigma_0);
}
