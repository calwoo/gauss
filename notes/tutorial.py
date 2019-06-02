import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-pastel")


def rbf_kernel(x, y, theta=1.0, sigma=1.0):
    """vectorized rbf kernel"""
    dist = np.sum(x**2, axis=1).reshape(-1,1) + np.sum(y**2, axis=1) - 2*np.dot(x, y.T)
    return sigma**2 * np.exp(-0.5 / theta**2 * dist)

X = np.arange(-5, 5, 0.2).reshape(-1,1)

mu = np.zeros(X.shape)
cov = rbf_kernel(X, X)

# sample
samples = np.random.multivariate_normal(mu.ravel(), cov, 3)

def gp_viz(xs, mu, cov, samples=None):
    """plots a visualization of the gaussian process
    using the given (already computed) parameters.
    Takes in samples of shape (num_samples, num_points)
    """
    if samples is None:
        samples = np.random.multivariate_normal(mu.ravel(), cov, 3)
    # ravel to 1d arrays
    xs = xs.ravel()
    mu = mu.ravel()

    num_samples = samples.shape[0]
    # plot mean
    plt.plot(xs, mu, label="mean")
    for i in range(num_samples):
        # plot sample
        sample = samples[i]
        plt.plot(xs, sample, linestyle="dashed", label="sample {}".format(i+1))

    # plot confidence interval
    deviations = np.sqrt(np.diag(cov))
    plt.fill_between(xs, mu+2*deviations, mu-2*deviations, alpha=0.15)
    
    plt.legend(loc="best")
    plt.show()
        
gp_viz(X, mu, cov, samples=samples)

def compute_posterior(x_test, x_train, y_train, theta=1.0, sigma=1.0, sigma_noise=1e-8):
    """
    computes the posterior distribution on the x_test points of the
    gaussian process with rbf kernel, given training data.
    INPUTS:
        x_test = test points, ndarray of shape (num_points, dim)
        x_train = training points, ndarray of shape (num_train, dim)
        y_train = training targets, (num_train, 1)
        kernel values theta, sigma, sigma_noise
    OUTPUT:
        mu_test, cov_test = (posterior) joint multivariate gaussian over test points
    """
    K_test_train = rbf_kernel(x_train, x_test, theta=theta, sigma=sigma)
    K_train = rbf_kernel(x_train, x_train, theta=theta, sigma=sigma) + sigma_noise**2 * np.identity(x_train.shape[0])
    K_test = rbf_kernel(x_test, x_test, theta=theta, sigma=sigma)

    # posterior gaussian equations
    mu_test = K_test_train.T @ np.linalg.inv(K_train) @ y_train
    cov_test = K_test - K_test_train.T @ np.linalg.inv(K_train) @ K_test_train
    return mu_test, cov_test


x_train = np.array([-4,-3,-2,-1,1]).reshape(-1,1)
y_train = np.sin(x_train)

mu_test, cov_test = compute_posterior(X, x_train, y_train, sigma_noise=0.4)

samples = np.random.multivariate_normal(mu_test.ravel(), cov_test, 5)
gp_viz(X, mu_test, cov_test, samples=samples)