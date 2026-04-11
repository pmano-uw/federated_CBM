data {
    int<lower=1> N;              // Total number of data
    real<lower=1> Delta;         // Lag time
    real<lower=0> sigma_0;       // Noise in Brownian motion

    vector[N] Delta_l_k;         // Observations' diff

    // EP Approximation parameters
    vector[2] mu_i;            // Mean of EP approx.
    matrix[2, 2] Sigma_i;      // Sigma of EP approx.
}
parameters {
    real<lower=0> mu;                     // mean of the central server
    real<lower=0> tau;           // variance of the central server
    real<lower=0> beta_i;                 // estimate of beta_i
}
transformed parameters {
    vector[2] phi;             // concat mu and tau
    real beta_i_scaled;         // mu array for each site
    real tau_scaled;

    phi[1] = mu;
    phi[2] = tau;

    beta_i_scaled = beta_i * Delta;
    tau_scaled = sqrt(tau);
}
model {
    target += multi_normal_lpdf(phi | mu_i, Sigma_i);
    target += normal_lpdf(beta_i | mu, tau_scaled);
    for (n in 1:N)
        target += normal_lpdf(Delta_l_k[n] | beta_i_scaled, sigma_0);
}