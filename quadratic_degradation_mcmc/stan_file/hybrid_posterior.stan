data {
    int<lower=1> N;              // Total number of data
    real<lower=1> Delta;         // Lag time
    real<lower=0> sigma_0;       // Noise in Brownian motion

    vector[N] Delta_l_k;         // Observations' diff
    vector[N] k;                 // Time

    // EP Approximation parameters
    vector[5] mu_i;              // Mean of EP approx. over phi = [mu, vech(Sigma)]
    matrix[5, 5] Sigma_i;        // Sigma of EP approx.

    // Population-level priors, matching the centralized model exactly
    vector[2] mu_mu;
    matrix[2, 2] Sigma_mu;
    real<lower=0> shape_Sigma;
    matrix[2, 2] nu_Sigma;
}
parameters {
    vector[2] mu;             // mean of the central server
    cov_matrix[2] Sigma;      // covariance of the central server
}
transformed parameters {
    vector[5] phi;               // concat mu and vech(Sigma)
    vector[2] t_k;

    vector[N] mu_scaled;              // mu array for each site
    vector[N] sigma_scaled;  // sigma array for each site

    phi[1:2] = mu;
    phi[3] = Sigma[1, 1];
    phi[4] = Sigma[1, 2];
    phi[5] = Sigma[2, 2];

    for (n in 1:N){
        t_k = [Delta, pow(Delta, 2) * k[n]]';
        mu_scaled[n] = mu' * t_k;
        sigma_scaled[n] = sqrt(pow(sigma_0, 2) + quad_form(Sigma, t_k));
    }
}
model {
    mu ~ multi_normal(mu_mu, Sigma_mu);
    Sigma ~ inv_wishart(shape_Sigma, nu_Sigma);

    target += multi_normal_lpdf(phi | mu_i, Sigma_i);
    for (n in 1:N)
        target += normal_lpdf(Delta_l_k[n] | mu_scaled[n], sigma_scaled[n]);
}
