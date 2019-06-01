import numpy as np
import matplotlib.pyplot as plt

from kernels import *
from utils import *



rbf = RBFKernel(theta=0.8, sigma=0.8)
periodic = PeriodicKernel(theta=0.8, periodicity=0.5, sigma=0.8)
linear = LinearKernel(b=0.8, c=0, sigma=0.3)


kernel_heatmap(linear)