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

def gp_viz(X, mu, cov, samples=None):
    """plots a visualization of the gaussian process
    using the given (already computed) parameters.
    Takes in samples of shape (num_samples, num_points)
    """
    if samples is None:
        samples = np.random.multivariate_normal(mu.ravel(), cov, 3)
    
    num_samples = samples.shape[0]
    # plot mean
    plt.plot(X, mu, label="mean")
    for i in range(num_samples):
        # plot sample
        sample = samples[i]
        plt.plot(X, sample, linestyle="dashed", label="sample {}".format(i+1))

    # plot confidence interval
    

    plt.legend(loc="best")
    plt.show()
        


gp_viz(X, mu, cov, samples=samples)