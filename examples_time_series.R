

# ==== GP time series with squared exponential kernel ======
# Load necessary libraries
library(tictoc)
library(cmdstanr)
library(ggplot2)
library(tidyverse)
library(magrittr)

# Set seed for reproducibility

# Set seed for reproducibility
set.seed(123)

# Number of data points
N <- 50

# Time points
t <- sort(runif(N, 0, 12))

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
  geom_line(aes(y = f_true), color = "blue", linetype = "dashed", linewidth = 1) +
  geom_point(aes(y = y, color = x), size = 2) +
  scale_color_gradient(low = "red", high = "green") +
  labs(x = "Month",
       y = "Network nestedness",
       color = "Functional space") +
  theme_minimal() +
  theme(axis.text = element_text(size = 10),
        axis.title = element_text(size = 15), 
        legend.position = 'top')

plot(x, y, xlab = 'Hypervolume', ylab = 'Cluestering', 
     col = 'lightblue', pch = 16, 
     cex.lab = 1.5)
abline(coef = c(-0.7, 0.5), lty = 3, col = 'tan1', lwd = 2)


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
    chains = 5, 
    parallel_chains = 5,
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
             
             mu1 <- 
               with(post, 
                    {
                      f[[x]] +
                        beta[[1]] * min(d$x)
                    })
             
             mu1 <- rnorm(1e3, mu1, post$sigma[[1]])
             
             mu2 <- 
               with(post, 
                    {
                      f[[x]] +
                        beta[[1]] * max(d$x)
                    })
             
             mu2 <- rnorm(1e3, mu2, post$sigma[[1]])
             
             tibble(x = d$t[x], 
                    li = c(min(mu), min(mu1), min(mu2)), 
                    ls = c(max(mu), max(mu1), max(mu2)), 
                    type = c('mean', 'min', 'max'))
             
           })

pred <- do.call('rbind', pred)

ggplot() +
  geom_ribbon(data = pred, aes(x, ymin = li, ymax = ls, 
                               fill = type), alpha = 0.2) +
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


# ==== GP time series with a circular variable (periodic kernel) ====

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
sigma_f_true <- 0.25  # Signal standard deviation
length_scale_true <- 5  # Length scale
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
    chains = 5, 
    parallel_chains = 5,
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


# ====== GP time series model stratified by a categorical variable =====


# Set seed for reproducibility
set.seed(123)

# Number of data points
N <- 30

# Circular time points (e.g., days of the year)
period <- 365
t <- sort(runif(N, 0, period))

# Categorical predictor (e.g., 2 categories)
C <- 2
cat <- sample(1:C, N, replace = TRUE)

# True parameters
sigma_f_true <- c(1.0, 1.5)  # Signal standard deviation for each category
length_scale_true <- c(5, 8)  # Length scale for each category
sigma_true <- 0.1  # Noise standard deviation

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
    if (cat[i] == cat[j]) {
      K[i, j] <- periodic_kernel(t[i], t[j], sigma_f_true[cat[i]], length_scale_true[cat[i]], period)
    }
  }
}

# Add a small value to the diagonal for numerical stability
K <- K + diag(1e-9, N)

# Simulate latent GP values
set.seed(123)
f_true <- MASS::mvrnorm(1, rep(0, N), K)

# Simulate observed data with noise
y <- f_true + rnorm(N, 0, sigma_true)

# Plot the simulated data
ggplot(data.frame(t = t, y = y, f_true = f_true, cat = as.factor(cat)), aes(x = t, y = y, color = cat)) +
  geom_line(aes(y = f_true, group = cat), linetype = "dashed", size = 1) +
  geom_point(size = 2) +
  labs(title = "Simulated Data with Circular Time Variable and Categorical Predictor",
       x = "Time (t)",
       y = "Observed Value (y)",
       color = "Category") +
  theme_minimal()



cat(file = 'GP_time_series_EG3.stan', 
    '
    data {
  int<lower=1> N;              // Number of data points
  vector[N] t;                 // Circular time points (e.g., days of the year)
  vector[N] y;                 // Observed values
  int<lower=1> C;              // Number of categories
  array[N] int<lower = 1, upper = C> cat; // Categorical predictor (e.g., 1, 2, ..., C)
  real<lower=0> period;        // Period of the circular time (e.g., 365 for days of the year)
  array[19] int y_cat1;
  array[11] int y_cat2;
}

parameters {
  real<lower=0> sigma;         // Standard deviation of the noise
  vector<lower=0>[C] length_scale;  // Length scale of the GP for each category
  vector<lower=0>[C] sigma_f;       // Signal standard deviation for each category
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
        if (cat[i] == cat[j]) {
          // Only compute covariance if the categories are the same
          real distance = abs(t[i] - t[j]);
          real periodic_distance = fmin(distance, period - distance); // Handle circularity
          K[i, j] = sigma_f[cat[i]]^2 * exp(-2 * square(sin(pi() * periodic_distance / period)) / square(length_scale[cat[i]]));
          K[j, i] = K[i, j];
        } else {
          // No covariance between different categories
          K[i, j] = 0;
          K[j, i] = 0;
        }
      }
      K[i, i] = sigma_f[cat[i]]^2 + 1e-9;  // Add a small value to the diagonal for numerical stability
    }
    K[N, N] = sigma_f[cat[N]]^2 + 1e-9;

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
  vector[19] y_predC1;  // Vector to store predictions for the observed data points
  vector[11] y_predC2;  // Vector to store predictions for the observed data points
  array[N] real y_pred;
  for (n in 1:19) {
    // Generate predictions by sampling from a normal distribution centered at `f[n]` with standard deviation `sigma`
    y_predC1[n] = normal_rng(f[y_cat1[n]], sigma);
  }
  
  for (n in 1:11) {
    // Generate predictions by sampling from a normal distribution centered at `f[n]` with standard deviation `sigma`
    y_predC2[n] = normal_rng(f[y_cat2[n]], sigma);
  }
  
  y_pred = normal_rng(f, sigma);
}
    ')

file <- paste0(getwd(), '/GP_time_series_EG3.stan')

fit <- cmdstan_model(file, compile = T)

which(stan_data$cat == 1)
which(stan_data$cat == 2)

stan_data <- list(
  N = N,
  y_cat1 = which(stan_data$cat == 1), 
  y_cat2 = which(stan_data$cat == 2),
  t = t,
  y = y,
  C = C,
  cat = cat,
  period = period
)

mod <- 
  fit$sample(
    data = stan_data, 
    iter_sampling = 2000, 
    iter_warmup = 1000, 
    thin = 10, 
    chains = 5, 
    parallel_chains = 5,
    seed = 123
  )

sum <- mod$summary()

sum |> print(n = 200)

post <- mod$draws(c('f', 'sigma'), format = 'df')

post <- 
  list(f = post[, grep('$f', colnames(post))], 
       sigma = post$sigma)

pred_cat <- mod$draws(c('y_predC1', 'y_predC2'), format = 'df')

pred_cat <- 
  list(cat1 = pred_cat[, grep('y_predC1', colnames(pred_cat))], 
       cat2 = pred_cat[, grep('y_predC2', colnames(pred_cat))])


pred_cat <- 
  lapply(1:2, FUN =
         function(i) {
           
           df <- pred_cat[[i]]
           
           d <- 
             lapply(seq_along(df), FUN = 
                      function(j) {
                        
                        tibble(li = quantile(df[[j]], 0.025), 
                               ls = quantile(df[[j]], 0.975))
                      })
           d <- do.call('rbind', d)
           d$cat <- as.factor(i)
           d$x <- stan_data$t[which(stan_data$cat == i)]
           d
           
         })

pred_cat <- do.call('rbind', pred_cat)


colnames(post)

d <- data.frame(t = t, y = y, f_true = f_true, cat = as.factor(cat))

ggplot() +
  geom_ribbon(data = pred_cat, 
              aes(x, ymin = li, ymax = ls, fill = cat), alpha = 0.3) +
  geom_line(data = d, 
            aes(x = t, color = cat, y = f_true, group = cat), linetype = "dashed", size = 1) +
  geom_point(data = d, 
             aes(x = t, color = cat, y = y, group = cat), 
             size = 2) +
  labs(title = "Simulated Data with Circular Time Variable and Categorical Predictor",
       x = "Time (t)",
       y = "Observed Value (y)",
       color = "Category") +
  theme_minimal()


# ======= GP time series model to forecasting =====

# Set seed for reproducibility
set.seed(123)

# Number of observed data points
N <- 100

# Circular time points (e.g., days of the year)
period <- 365
t <- sort(runif(N, 0, period))

# Categorical predictor (e.g., 2 categories)
C <- 2
cat <- sample(1:C, N, replace = TRUE)

# True parameters
sigma_f_true <- c(1.0, 1.5)  #// Signal standard deviation for each category
length_scale_true <- c(100.0, 150.0)  #// Length scale for each category

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
    if (cat[i] == cat[j]) {
      K[i, j] <- periodic_kernel(t[i], t[j], sigma_f_true[cat[i]], length_scale_true[cat[i]], period)
    }
  }
}

# Add a small value to the diagonal for numerical stability
K <- K + diag(1e-9, N)

# Simulate latent GP values
f_true <- MASS::mvrnorm(1, rep(0, N), K)

# Simulate observed counts using a Poisson distribution
y <- rpois(N, exp(f_true))

# Future time points for forecasting
N_new <- 50
t_new <- seq(max(t) + 1, max(t) + N_new, length.out = N_new)
cat_new <- sample(1:C, N_new, replace = TRUE)

# Plot the simulated data
ggplot(data.frame(t = t, y = y, f_true = f_true, cat = as.factor(cat)), aes(x = t, y = y, color = cat)) +
  geom_line(aes(y = exp(f_true), group = cat), linetype = "dashed", size = 1) +
  geom_point(size = 2) +
  labs(title = "Simulated Data with Circular Time Variable and Categorical Predictor",
       x = "Time (t)",
       y = "Observed Counts (y)",
       color = "Category") +
  theme_minimal()


cat(file = 'GP_time_series_EG4.stan', 
    '
    data {
  int<lower=1> N;              // Number of observed data points
  int<lower=1> N_new;          // Number of future time points for forecasting
  vector[N] t;                 // Observed time points
  vector[N_new] t_new;         // Future time points for forecasting
  int y[N];                    // Observed counts (Poisson-distributed)
  int<lower=1> C;              // Number of categories
  int<lower=1, upper=C> cat[N]; // Categorical predictor for observed data
  int<lower=1, upper=C> cat_new[N_new]; // Categorical predictor for future data
  real<lower=0> period;        // Period of the circular time (e.g., 365 for days of the year)
}

parameters {
  vector<lower=0>[C] length_scale;  // Length scale of the GP for each category
  vector<lower=0>[C] sigma_f;       // Signal standard deviation for each category
  vector[N] eta;               // Latent variables for the GP
}

transformed parameters {
  vector[N] f;  // Latent GP function values (log of the Poisson rate)
  {
    matrix[N, N] K;  // Covariance matrix for the observed data
    matrix[N, N] L_K;  // Cholesky decomposition of the covariance matrix

    // Construct the covariance matrix using the periodic kernel
    for (i in 1:(N-1)) {
      for (j in (i+1):N) {
        if (cat[i] == cat[j]) {
          // Only compute covariance if the categories are the same
          real distance = fabs(t[i] - t[j]);
          real periodic_distance = fmin(distance, period - distance); // Handle circularity
          K[i, j] = sigma_f[cat[i]]^2 * exp(-2 * square(sin(pi() * periodic_distance / period)) / square(length_scale[cat[i]]));
          K[j, i] = K[i, j];
        } else {
          // No covariance between different categories
          K[i, j] = 0;
          K[j, i] = 0;
        }
      }
      K[i, i] = sigma_f[cat[i]]^2 + 1e-9;  // Add a small value to the diagonal for numerical stability
    }
    K[N, N] = sigma_f[cat[N]]^2 + 1e-9;

    // Perform Cholesky decomposition of the covariance matrix
    L_K = cholesky_decompose(K);

    // Transform the latent variables `eta` into the GP values `f`
    f = L_K * eta;
  }
}

model {
  // Priors for the parameters
  length_scale ~ inv_gamma(5, 5);  // Inverse gamma prior for the length scale
  sigma_f ~ cauchy(0, 1);  // Cauchy prior for the signal standard deviation
  eta ~ normal(0, 1);  // Standard normal prior for the latent variables

  // Likelihood: observed counts `y` are Poisson-distributed with rate exp(f)
  y ~ poisson_log(f);
}

generated quantities {
  int y_pred[N];  // Predictions for observed data points
  vector[N_new] f_new;  // Latent GP function values for future time points
  int y_new[N_new];  // Predictions for future time points

  // Compute the latent GP values for future time points
  {
    matrix[N_new, N] K_new;  // Covariance matrix between observed and future time points
    matrix[N_new, N_new] K_new_new;  // Covariance matrix for future time points
    matrix[N_new, N_new] L_K_new_new;  // Cholesky decomposition of the future covariance matrix

    // Compute the covariance between observed and future time points
    for (i in 1:N_new) {
      for (j in 1:N) {
        if (cat_new[i] == cat[j]) {
          real distance = fabs(t_new[i] - t[j]);
          real periodic_distance = fmin(distance, period - distance); // Handle circularity
          K_new[i, j] = sigma_f[cat_new[i]]^2 * exp(-2 * square(sin(pi() * periodic_distance / period)) / square(length_scale[cat_new[i]]));
        } else {
          K_new[i, j] = 0;
        }
      }
    }

    // Compute the covariance for future time points
    for (i in 1:(N_new-1)) {
      for (j in (i+1):N_new) {
        if (cat_new[i] == cat_new[j]) {
          real distance = fabs(t_new[i] - t_new[j]);
          real periodic_distance = fmin(distance, period - distance); // Handle circularity
          K_new_new[i, j] = sigma_f[cat_new[i]]^2 * exp(-2 * square(sin(pi() * periodic_distance / period)) / square(length_scale[cat_new[i]]));
          K_new_new[j, i] = K_new_new[i, j];
        } else {
          K_new_new[i, j] = 0;
          K_new_new[j, i] = 0;
        }
      }
      K_new_new[i, i] = sigma_f[cat_new[i]]^2 + 1e-9;  // Add a small value to the diagonal for numerical stability
    }
    K_new_new[N_new, N_new] = sigma_f[cat_new[N_new]]^2 + 1e-9;

    // Perform Cholesky decomposition of the future covariance matrix
    L_K_new_new = cholesky_decompose(K_new_new);

    // Generate latent GP values for future time points
    f_new = K_new * inv(K) * f;
  }

  // Generate predictions for observed and future time points
  for (n in 1:N) {
    y_pred[n] = poisson_log_rng(f[n]);
  }
  for (n in 1:N_new) {
    y_new[n] = poisson_log_rng(f_new[n]);
  }
}
    '
    )

# ===== GP time series model combining kernel ==== 

cat(file = 'GP_time_series_EG5.stan', 
    '
    data {
  int<lower=1> N;              // Number of months (e.g., 24 for 2 years)
  vector[N] t;                 // Time points (e.g., 1, 2, ..., 24)
  vector[N] y;                 // Observed weather data (e.g., temperature)
  real<lower=0> period;        // Period of the circular time (e.g., 12 for monthly data)
}

parameters {
  real<lower=0> sigma;         // Standard deviation of the noise
  real<lower=0> length_scale_periodic;  // Length scale for the periodic component
  real<lower=0> sigma_f_periodic;       // Signal standard deviation for the periodic component
  real<lower=0> length_scale_se;        // Length scale for the squared exponential component
  real<lower=0> sigma_f_se;             // Signal standard deviation for the squared exponential component
  vector[N] eta;               // Latent variables for the GP
}

transformed parameters {
  vector[N] f;  // Latent GP function values
  {
    matrix[N, N] K_periodic;  // Covariance matrix for the periodic component
    matrix[N, N] K_se;        // Covariance matrix for the squared exponential component
    matrix[N, N] K;           // Combined covariance matrix
    matrix[N, N] L_K;         // Cholesky decomposition of the combined covariance matrix

    // Construct the periodic covariance matrix
    for (i in 1:(N-1)) {
      for (j in (i+1):N) {
        real distance = abs(t[i] - t[j]);
        real periodic_distance = fmin(distance, period - distance); // Handle circularity
        K_periodic[i, j] = sigma_f_periodic^2 * exp(-2 * square(sin(pi() * periodic_distance / period)) / square(length_scale_periodic));
        K_periodic[j, i] = K_periodic[i, j];
      }
      K_periodic[i, i] = sigma_f_periodic^2 + 1e-9;  // Add a small value to the diagonal for numerical stability
    }
    K_periodic[N, N] = sigma_f_periodic^2 + 1e-9;

    // Construct the squared exponential covariance matrix
    for (i in 1:(N-1)) {
      for (j in (i+1):N) {
        K_se[i, j] = sigma_f_se^2 * exp(-0.5 * square(t[i] - t[j]) / square(length_scale_se));
        K_se[j, i] = K_se[i, j];
      }
      K_se[i, i] = sigma_f_se^2 + 1e-9;
    }
    K_se[N, N] = sigma_f_se^2 + 1e-9;

    // Combine the two covariance matrices
    K = K_periodic + K_se;

    // Perform Cholesky decomposition of the combined covariance matrix
    L_K = cholesky_decompose(K);

    // Transform the latent variables `eta` into the GP values `f`
    f = L_K * eta;
  }
}

model {
  // Priors for the parameters
  sigma ~ cauchy(0, 1);  // Cauchy prior for the noise standard deviation
  length_scale_periodic ~ inv_gamma(5, 5);  // Inverse gamma prior for the periodic length scale
  sigma_f_periodic ~ cauchy(0, 1);  // Cauchy prior for the periodic signal standard deviation
  length_scale_se ~ inv_gamma(5, 5);  // Inverse gamma prior for the squared exponential length scale
  sigma_f_se ~ cauchy(0, 1);  // Cauchy prior for the squared exponential signal standard deviation
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

# Number of months (2 years)
N <- 24

# Time points (1, 2, ..., 24)
t <- 1:N

# Period (12 months)
period <- 12

# True parameters
sigma_f_periodic_true <- 1.0 # // Signal standard deviation for the periodic component
length_scale_periodic_true <- 1.0  #// Length scale for the periodic component
sigma_f_se_true <- 0.5  #// Signal standard deviation for the squared exponential component
length_scale_se_true <- 10.0  #// Length scale for the squared exponential component
sigma_true <- 0.1  # // Noise standard deviation

# Periodic kernel function
periodic_kernel <- function(t1, t2, sigma_f, length_scale, period) {
  distance <- abs(t1 - t2)
  periodic_distance <- pmin(distance, period - distance)
  sigma_f^2 * exp(-2 * sin(pi * periodic_distance / period)^2 / length_scale^2)
}

# Squared exponential kernel function
se_kernel <- function(t1, t2, sigma_f, length_scale) {
  sigma_f^2 * exp(-0.5 * (t1 - t2)^2 / length_scale^2)
}

# Generate the covariance matrices
K_periodic <- matrix(0, N, N)
K_se <- matrix(0, N, N)
for (i in 1:N) {
  for (j in 1:N) {
    K_periodic[i, j] <- periodic_kernel(t[i], t[j], sigma_f_periodic_true, length_scale_periodic_true, period)
    K_se[i, j] <- se_kernel(t[i], t[j], sigma_f_se_true, length_scale_se_true)
  }
}

# Combine the covariance matrices
K <- K_periodic + K_se

# Add a small value to the diagonal for numerical stability
K <- K + diag(1e-9, N)

# Simulate latent GP values
f_true <- MASS::mvrnorm(1, rep(0, N), K)

# Simulate observed weather data with noise
y <- f_true + rnorm(N, 0, sigma_true)

# Plot the simulated data
ggplot(data.frame(t = t, y = y, f_true = f_true), aes(x = t)) +
  geom_line(aes(y = f_true), color = "blue", linetype = "dashed", size = 1) +
  geom_point(aes(y = y), color = "red", size = 2) +
  labs(title = "Simulated Monthly Weather Data for Two Consecutive Years",
       x = "Time (t)",
       y = "Observed Weather (y)") +
  theme_minimal()



file <- paste0(getwd(), '/GP_time_series_EG5.stan')

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
    chains = 5, 
    parallel_chains = 5,
    seed = 123
  )

mod$summary() |> print(n = 200)

pred <- mod$draws('y_pred', format = 'df')

pred <- pred[, grep('y_pred', colnames(pred))]

pred <- 
  do.call('rbind', 
          lapply(seq_along(pred), FUN = 
                   function(x) {
                     tibble(x = x, 
                            li = quantile(pred[[x]], 0.025), 
                            ls = quantile(pred[[x]], 0.975))
                   }))

d <- data.frame(t = t, y = y, f_true = f_true)

ggplot() +
  geom_ribbon(data = pred, aes(x, ymin = li, ymax = ls), alpha = 0.3) +
  geom_line(data = d, aes(x = t, y = f_true), color = "blue", linetype = "dashed", size = 1) +
  geom_point(data = d, aes(x = t, y = y), color = "red", size = 2) +
  labs(title = "Simulated Monthly Weather Data for Two Consecutive Years",
       x = "Time (t)",
       y = "Observed Weather (y)") +
  theme_minimal()















# ====== GP ===

# Set seed for reproducibility
set.seed(123)

# Number of months (2 years)
M <- 24

# Number of observations per month (e.g., daily data)
obs_per_month <- 10

# Total number of observations
N <- M * obs_per_month

# Month index for each observation
month <- rep(1:M, each = obs_per_month)

# Period (12 months)
period <- 12

# True parameters
sigma_f_true <- 1.0  #// Signal standard deviation for the GP
length_scale_true <- 1.0  #// Length scale for the GP
sigma_true <- 0.1  #// Noise standard deviation

# Periodic kernel function
periodic_kernel <- function(t1, t2, sigma_f, length_scale, period) {
  distance <- abs(t1 - t2)
  periodic_distance <- pmin(distance, period - distance)
  sigma_f^2 * exp(-2 * sin(pi * periodic_distance / period)^2 / length_scale^2)
}

# Generate the covariance matrix
K <- matrix(0, M, M)
for (i in 1:(M-1)) {
  for (j in (i+1):M) {
    K[i, j] <- periodic_kernel(i, j, sigma_f_true, length_scale_true, period)
    K[j, i] <- K[i, j];
  }
  K[i, i] <- sigma_f_true^2 + 1e-9;
}
K[M, M] <- sigma_f_true^2 + 1e-9;

# Simulate latent GP values
f_true <- MASS::mvrnorm(1, rep(0, M), K)

# Simulate observed weather data with noise
y <- f_true[month] + rnorm(N, 0, sigma_true)

# Plot the simulated data
ggplot(data.frame(t = 1:N, y = y, f_true = f_true[month], month = as.factor(month)), aes(x = t, y = y, color = month)) +
  geom_line(aes(y = f_true), color = "blue", linetype = "dashed", size = 1) +
  geom_point(size = 2) +
  labs(title = "Simulated Monthly Weather Data",
       x = "Time (t)",
       y = "Observed Weather (y)",
       color = "Month") +
  theme_minimal()


cat(file = 'GP_time_series_EG5.stan', 
    '
    data {
  int<lower=1> N;              // Total number of observations
  int<lower=1> M;              // Number of months
  array[N] int month;          // Month index for each observation (1, 2, ..., M)
  vector[N] y;                 // Observed weather data (e.g., temperature)
  real<lower=0> period;        // Period of the circular time (e.g., 12 for monthly data)
}

parameters {
  real<lower=0> sigma;         // Standard deviation of the noise
  real<lower=0> length_scale;  // Length scale for the GP
  real<lower=0> sigma_f;       // Signal standard deviation for the GP
  vector[M] eta;               // Latent variables for the GP
}

transformed parameters {
  vector[M] f;  // Latent GP function values (monthly trends)
  {
    matrix[M, M] K;  // Covariance matrix for the GP
    matrix[M, M] L_K;  // Cholesky decomposition of the covariance matrix

    // Construct the covariance matrix using the periodic kernel
    for (i in 1:(M-1)) {
      for (j in (i+1):M) {
        real distance = abs(i - j);
        real periodic_distance = fmin(distance, period - distance); // Handle circularity
        K[i, j] = sigma_f^2 * exp(-2 * square(sin(pi() * periodic_distance / period)) / square(length_scale));
        K[j, i] = K[i, j];
      }
      K[i, i] = sigma_f^2 + 1e-9;  // Add a small value to the diagonal for numerical stability
    }
    K[M, M] = sigma_f^2 + 1e-9;

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

  // Likelihood: observed data `y` is normally distributed around the GP values `f[month]`
  y ~ normal(f[month], sigma);
}

generated quantities {
  vector[N] y_pred;  // Vector to store predictions for the observed data points
  for (n in 1:N) {
    // Generate predictions by sampling from a normal distribution centered at `f[month[n]]` with standard deviation `sigma`
    y_pred[n] = normal_rng(f[month[n]], sigma);
  }
}
    ')

stan_data <- list(
  N = N,
  M = M,
  month = month,
  y = y,
  period = period
)

file <- paste0(getwd(), '/GP_time_series_EG5.stan')

fit <- cmdstan_model(file, compile = T)

mod <- 
  fit$sample(
    data = stan_data, 
    iter_sampling = 2000, 
    iter_warmup = 1000, 
    thin = 10, 
    chains = 5, 
    parallel_chains = 5,
    seed = 123
  )


sum <- mod$summary()


post <- mod$draws(c('f', 'sigma'), format = 'df')

post <- 
  list(f = post[, grep('f', colnames(post))], 
       sigma = post[, grep('sigma', colnames(post))])


pred_y <- mod$draws('y_pred', format = 'df')

pred_y <- pred_y[, grep('y_pred', colnames(pred_y))]

pred <- 
  lapply(seq_along(pred_y), FUN = 
           function(x) {
             
             mu <- pred_y[[x]]
             
             tibble(x = x, 
                    li = quantile(mu, 0.025), 
                    ls = quantile(mu, 0.975))
             
           })

pred <- do.call('rbind', pred)

d <- data.frame(t = 1:N, y = y, f_true = f_true[month], month = as.factor(month))

ggplot() +
  geom_ribbon(data = pred, aes(x, ymin = li, ymax = ls), alpha = 0.3) +
  geom_line(data = d, aes(x = t, y = f_true), color = "blue", linetype = "dashed", size = 1) +
  geom_point(data = d, aes(t, y), size = 2) +
  labs(title = "Simulated Monthly Weather Data",
       x = "Time (t)",
       y = "Observed Weather (y)",
       color = "Month") +
  theme_minimal()





