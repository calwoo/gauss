import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-pastel")


class GP:
    """
    implementation of a gaussian process
    """
    def __init__(self, kernel, mean=None):
        if mean is None:
            self.mean = lambda x: 0
        else:
            self.mean = mean
        self.kernel = kernel
        self.x_train = None
        self.y_train = None

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.K_train = self.kernel.eval(self.x_train, self.x_train)

    def predict(self, x_test, noise=0.0, viz=False):
        K_test_train = self.kernel.eval(self.x_train, x_test)
        K_train = self.K_train + noise**2 * np.identity(self.x_train.shape[0])
        K_test = self.kernel.eval(x_test, x_test)

        # posterior gaussian equations
        mu_test = K_test_train.T @ np.linalg.inv(K_train) @ self.y_train
        cov_test = K_test - K_test_train.T @ np.linalg.inv(K_train) @ K_test_train  
        
        if viz:
            self.viz(x_test, mu_test, cov_test)
        
        return mu_test, cov_test

    def viz(self, xs, mu, cov, samples=None):
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
        
        if num_samples < 5:
            plt.legend(loc="best")
        plt.show()
