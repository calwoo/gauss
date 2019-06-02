import numpy as np
from scipy.special import gamma, kv

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

    
class PeriodicKernel(Kernel):
    """periodic kernel from
        https://distill.pub/2019/visual-exploration-gaussian-processes"""
    def __init__(self, theta, periodicity, sigma=1.0):
        super(PeriodicKernel, self).__init__(sigma)
        self.theta = theta
        self.periodicity = periodicity
        
    def eval(self, x, y):
        dist = np.sqrt((x-y)**2)
        logit = -2*np.sin(np.pi * dist / self.periodicity)**2 / self.theta**2
        return self.sigma * np.exp(logit)


class LinearKernel(Kernel):
    """linear polynomial kernel"""
    def __init__(self, b=0, c=0, sigma=1.0):
        super(LinearKernel, self).__init__(sigma)
        self.b = b
        self.c = c

    def eval(self, x, y):
        return self.b**2 + self.sigma**2 * np.dot(x-self.c, y-self.c)

class MaternKernel(Kernel):
    """Matern kernel from
        https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function"""
    def __init__(self, rho, nu, sigma=1.0):
        super(MaternKernel, self).__init__(sigma)
        self.rho = rho
        self.nu = nu

    def eval(self, x, y):
        dist = np.sqrt(2 * np.nu * np.sum((x-y)**2)) / self.rho
        coef = 2**(1-self.nu) / gamma(self.nu)
        return coef * dist**self.nu * kv(self.nu, dist)
        
