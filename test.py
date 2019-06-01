import numpy as np
import matplotlib.pyplot as plt

from kernels import *
from utils import *


kern = RBFKernel(theta=0.8, sigma=0.8)

x = np.array([1,2])
y = np.array([0,0])
print(kern.eval(x,y))



kernel_heatmap(kern)