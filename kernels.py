import numpy as np


class Kernel:
    """initializes default L2 distance kernel"""
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def eval(self, x, y):
        return self.sigma * np.sum((x-y)**2)



class RBFKernel(Kernel):
    """radial basis function (gaussian) kernel"""
    def __init__(self, theta, sigma=1.0):
        super(RBFKernel, self).__init__()
        self.theta = theta

    def eval(self, x, y):
        dist = np.sum((x-y)**2)
        return self.sigma * np.exp(-dist / (2*self.theta**2))