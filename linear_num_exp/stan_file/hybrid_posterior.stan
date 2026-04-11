data {
    int<lower=1> N;              // Total number of data
    real<lower=1> Delta;         // Lag time
    real<lower=0> sigma_0;       // Noise in Brownian motion

    vector[N] Delta_l_k;         // Observations' diff

    // EP Approximation parameters
    vector[2] mu_i;              // Mean of EP approx.
    matrix[2, 2] Sigma_i;        // Sigma of EP approx.
}
parameters {
    real<lower=0> mu;                     // mean of the central server
    real<lower=0> tau;           // variance of the central server
}
transformed parameters {
    vector[2] phi;               // concat mu and tau
    real mu_scaled;              // mu array for each site
    real sigma_scaled;  // sigma array for each site

    phi[1] = mu;
    phi[2] = tau;

    mu_scaled = mu * Delta;
    sigma_scaled = sqrt(pow(sigma_0, 2) + pow(tau, 2) * pow(Delta, 2));
}
model {
    target += multi_normal_lpdf(phi | mu_i, Sigma_i);
    for (n in 1:N)
        target += normal_lpdf(Delta_l_k[n] | mu_scaled, sigma_scaled);
}