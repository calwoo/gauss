import numpy as np
import matplotlib.pyplot as plt

from kernels import *
from utils import *
from gp import *



rbf = RBFKernel(theta=0.8, sigma=0.8)
periodic = PeriodicKernel(theta=0.8, periodicity=0.5, sigma=0.8)
linear = LinearKernel(b=0.8, c=0, sigma=0.3)


# kernel_heatmap(linear)

### regression
"""
given a model y=f(x)+noise where f(x)=-cos(2*pi*x)+0.5*sin(6*pi*x), noise~N(0,0.01),
can we use our gaussian process to sample?
"""
f = lambda x: -np.cos(2*np.pi*x) + 0.5*np.sin(6*np.pi*x)
x_train = np.linspace(0.05, 0.95, 10)[:,None]
y_train = f(x_train) + np.random.normal(0, 0.1, (10,1))

# plot
plt.plot(x_train, y_train, "kx")
plt.show()

# set up gaussian process
rbf_kernel = RBFKernel(theta=0.1, sigma=1.0)
gp = GP(rbf_kernel)

# train gp model on training points
gp.train(x_train, y_train)

# predict on new test points
x_test = np.linspace(-0.05, 1.05, 100)[:, None]
gp.predict(x_test, viz=True)