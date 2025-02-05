

# ==== time series with squared exponential kernel ======
# Load necessary libraries
library(tictoc)
library(cmdstanr)
library(ggplot2)

# Set seed for reproducibility

# Set seed for reproducibility
set.seed(123)

# Number of data points
N <- 50

# Time points
t <- sort(runif(N, 0, 10))

# Additional predictor
x <- rnorm(N, mean = 0, sd = 1)

# True parameters
sigma_f_true <- 1.0  # Signal standard deviation
length_scale_true <- 2.0  # Length scale
sigma_true <- 0.1  # Noise standard deviation
beta_true <- 0.5  # Coefficient for the additional predictor

# Generate the covariance matrix
squared_exp_kernel <- function(t1, t2, sigma_f, length_scale) {
  sigma_f^2 * exp(-0.5 * (t1 - t2)^2 / length_scale^2)
}

K <- matrix(0, N, N)
for (i in 1:N) {
  for (j in 1:N) {
    K[i, j] <- squared_exp_kernel(t[i], t[j], sigma_f_true, length_scale_true)
  }
}

# Add a small value to the diagonal for numerical stability
K <- K + diag(1e-9, N)

# Simulate latent GP values
set.seed(123)
f_true <- MASS::mvrnorm(1, rep(0, N), K)

# Simulate observed data with noise and predictor effect
y <- f_true + beta_true * x + rnorm(N, 0, sigma_true)

# Plot the simulated data
ggplot(data.frame(t = t, y = y, f_true = f_true, x = x), aes(x = t)) +
  geom_line(aes(y = f_true), color = "blue", linetype = "dashed", size = 1) +
  geom_point(aes(y = y, color = x), size = 2) +
  scale_color_gradient(low = "red", high = "green") +
  labs(title = "Simulated Data with Additional Predictor",
       x = "Time (t)",
       y = "Observed Value (y)",
       color = "Predictor (x)") +
  theme_minimal()

stan_data <- list(
  N = N,
  t = t,
  y = y,
  x = x
)

cat(file = 'GP_time_series_EG.stan', 
    '
    data {
  int<lower=1> N;       // Number of data points
  vector[N] t;          // Time points
  vector[N] y;          // Observed values
  vector[N] x;          // Additional predictor
}

parameters {
  real<lower=0> sigma;  // Standard deviation of the noise
  real<lower=0> length_scale;  // Length scale of the GP
  real<lower=0> sigma_f;       // Signal standard deviation
  real beta;                   // Coefficient for the additional predictor
  vector[N] eta;               // Latent GP values
}

transformed parameters {
  vector[N] f;  // Latent GP function values
  {
    matrix[N, N] K;  // Covariance matrix for the GP
    matrix[N, N] L_K;  // Cholesky decomposition of the covariance matrix

    // Construct the covariance matrix using the squared exponential kernel
    for (i in 1:(N-1)) {
      for (j in (i+1):N) {
        // Compute the covariance between time points t[i] and t[j]
        K[i, j] = sigma_f^2 * exp(-0.5 * square(t[i] - t[j]) / square(length_scale));
        // The covariance matrix is symmetric, so fill the lower triangle
        K[j, i] = K[i, j];
      }
      // Add a small value to the diagonal for numerical stability
      K[i, i] = sigma_f^2 + 1e-9;
    }
    // Ensure the last diagonal element is also stable
    K[N, N] = sigma_f^2 + 1e-9;

    // Perform Cholesky decomposition of the covariance matrix
    L_K = cholesky_decompose(K);

    // Transform the latent variables `eta` into the GP values `f`
    f = L_K * eta;
  }
}

model {
  // Priors for the parameters
  sigma ~ cauchy(0, 1);  // Cauchy prior for the noise standard deviation
  length_scale ~ inv_gamma(5, 5);  // Inverse gamma prior for the length scale
  sigma_f ~ cauchy(0, 1);  // Cauchy prior for the signal standard deviation
  beta ~ normal(0, 1);  // Normal prior for the predictor coefficient
  eta ~ normal(0, 1);  // Standard normal prior for the latent variables

  // Likelihood: observed data `y` is normally distributed around the GP values `f` plus the predictor effect
  y ~ normal(f + beta * x, sigma);
}

generated quantities {
  vector[N] y_pred;  // Vector to store predictions for the observed data points
  for (n in 1:N) {
    // Generate predictions by sampling from a normal distribution centered at `f[n] + beta * x[n]` with standard deviation `sigma`
    y_pred[n] = normal_rng(f[n] + beta * x[n], sigma);
  }
}
    ')


file <- paste0(getwd(), '/GP_time_series_EG.stan')

fit <- cmdstan_model(file, compile = T)

mod <- 
  fit$sample(
    data = stan_data, 
    iter_sampling = 2000, 
    iter_warmup = 1000, 
    thin = 10, 
    chains = 10, 
    parallel_chains = 3,
    seed = 123
  )

sum <- mod$summary()

d <- data.frame(t = t, y = y, f_true = f_true, x = x)

sum <- sum[grep('^f', sum$variable), ]

sum$x <- d$t

post <- mod$draws(c('f', 'beta', 'sigma'), format = 'df')

post <- 
  list(f = post[, grep('f', colnames(post))], 
       beta = post[, grep('beta', colnames(post))], 
       sigma = post[, grep('sigma', colnames(post))])

pred <- 
  lapply(1:50, FUN = 
           function(x) {
             
             mu <- 
               with(post, 
                    {
                      f[[x]] +
                        beta[[1]] * mean(d$x)
                    })
             
             mu <- rnorm(1e3, mu, post$sigma[[1]])
             
             tibble(x = d$t[x], 
                    li = min(mu), 
                    ls = max(mu))
             
           })

pred <- do.call('rbind', pred)

ggplot() +
  geom_ribbon(data = pred, aes(x, ymin = li, ymax = ls), alpha = 0.2) +
  geom_line(data = d, aes(x= t, y = f_true), color = "blue", linetype = "dashed", size = 1) +
  geom_errorbar(data = sum, 
                aes(x, ymin = `q5`, ymax = `q95`), width = 0) +
  geom_point(data = d, aes(x= t,y = y, color = x), size = 2) +
  scale_color_gradient(low = "red", high = "green") +
  labs(title = "Simulated Data with Additional Predictor",
       x = "Time (t)",
       y = "Observed Value (y)",
       color = "Predictor (x)") +
  theme_minimal()


# ==== time series with a circular variable (periodic kernel) ====

cat(file = 'GP_time_series_EG2.stan', 
    '
    data {
  int<lower=1> N;       // Number of data points
  vector[N] t;          // Circular time points (e.g., days of the year)
  vector[N] y;          // Observed values
  real<lower=0> period; // Period of the circular time (e.g., 365 for days of the year)
}

parameters {
  real<lower=0> sigma;  // Standard deviation of the noise
  real<lower=0> length_scale;  // Length scale of the GP
  real<lower=0> sigma_f;       // Signal standard deviation
  vector[N] eta;               // Latent variables for the GP
}

transformed parameters {
  vector[N] f;  // Latent GP function values
  {
    matrix[N, N] K;  // Covariance matrix for the GP
    matrix[N, N] L_K;  // Cholesky decomposition of the covariance matrix

    // Construct the covariance matrix using the periodic kernel
    for (i in 1:(N-1)) {
      for (j in (i+1):N) {
        // Periodic kernel: accounts for circular time
        real distance = abs(t[i] - t[j]);
        real periodic_distance = fmin(distance, period - distance); // Handle circularity
        K[i, j] = sigma_f^2 * exp(-2 * square(sin(pi() * periodic_distance / period)) / square(length_scale));
        K[j, i] = K[i, j];
      }
      K[i, i] = sigma_f^2 + 1e-9;  // Add a small value to the diagonal for numerical stability
    }
    K[N, N] = sigma_f^2 + 1e-9;

    // Perform Cholesky decomposition of the covariance matrix
    L_K = cholesky_decompose(K);

    // Transform the latent variables `eta` into the GP values `f`
    f = L_K * eta;
  }
}

model {
  // Priors for the parameters
  sigma ~ cauchy(0, 1);  // Cauchy prior for the noise standard deviation
  length_scale ~ inv_gamma(5, 5);  // Inverse gamma prior for the length scale
  sigma_f ~ cauchy(0, 1);  // Cauchy prior for the signal standard deviation
  eta ~ normal(0, 1);  // Standard normal prior for the latent variables

  // Likelihood: observed data `y` is normally distributed around the GP values `f`
  y ~ normal(f, sigma);
}

generated quantities {
  vector[N] y_pred;  // Vector to store predictions for the observed data points
  for (n in 1:N) {
    // Generate predictions by sampling from a normal distribution centered at `f[n]` with standard deviation `sigma`
    y_pred[n] = normal_rng(f[n], sigma);
  }
}
    ')

# Set seed for reproducibility
set.seed(123)

# Number of data points
N <- 100

# Circular time points (e.g., days of the year)
period <- 365
t <- sort(runif(N, 0, period))

# True parameters
sigma_f_true <- 0.5  # Signal standard deviation
length_scale_true <- 10  # Length scale
sigma_true <- 0.05  # Noise standard deviation

# Periodic kernel function
periodic_kernel <- function(t1, t2, sigma_f, length_scale, period) {
  distance <- abs(t1 - t2)
  periodic_distance <- pmin(distance, period - distance)
  sigma_f^2 * exp(-2 * sin(pi * periodic_distance / period)^2 / length_scale^2)
}

# Generate the covariance matrix
K <- matrix(0, N, N)
for (i in 1:N) {
  for (j in 1:N) {
    K[i, j] <- periodic_kernel(t[i], t[j], sigma_f_true, length_scale_true, period)
  }
}

# Add a small value to the diagonal for numerical stability
K <- K + diag(1e-9, N)

# Simulate latent GP values
f_true <- MASS::mvrnorm(1, rep(0, N), K)

# Simulate observed data with noise
y <- f_true + rnorm(N, 0, sigma_true)

# Plot the simulated data
ggplot(data.frame(t = t, y = y, f_true = f_true), aes(x = t)) +
  geom_line(aes(y = f_true), color = "blue", linetype = "dashed", size = 1) +
  geom_point(aes(y = y), color = "red", size = 2) +
  labs(title = "Simulated Data with Circular Time Variable",
       x = "Time (t)",
       y = "Observed Value (y)") +
  theme_minimal()



file <- paste0(getwd(), '/GP_time_series_EG2.stan')

fit <- cmdstan_model(file, compile = T)

stan_data <- list(
  N = N,
  t = t,
  y = y,
  period = period
)

mod <- 
  fit$sample(
    data = stan_data, 
    iter_sampling = 2000, 
    iter_warmup = 1000, 
    thin = 10, 
    chains = 3, 
    parallel_chains = 3,
    seed = 123
  )

sum <- mod$summary()


post <- mod$draws(c('f', 'sigma'), format = 'df')

post <- 
  list(f = post[, grep('f', colnames(post))], 
       sigma = post[, grep('sigma', colnames(post))])

pred <- 
  lapply(1:100, FUN = 
           function(x) {
             
             mu <- 
               with(post, 
                    {
                      f[[x]] 
                    })
             
             mu <- rnorm(1e3, mu, post$sigma[[1]])
             
             tibble(x = stan_data$t[x], 
                    li = quantile(mu, 0.025), 
                    ls = quantile(mu, 0.975))
             
           })

pred <- do.call('rbind', pred)

ggplot() +
  geom_ribbon(data = pred, aes(x, ymin = li, ymax = ls), alpha = 0.2) +
  geom_line(data = data.frame(t = t, y = y, f_true = f_true),
            aes(x = t, y = f_true), color = "blue", linetype = "dashed", size = 1) +
  geom_point(data = data.frame(t = t, y = y, f_true = f_true), 
             aes(x = t, y = y), color = "red", size = 2) +
  labs(title = "Simulated Data with Circular Time Variable",
       x = "Time (t)",
       y = "Observed Value (y)") +
  theme_minimal()

