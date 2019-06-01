import numpy as np
import matplotlib.pyplot as plt

from kernels import *



kern = Kernel()

x = np.array([1,2])
y = np.array([0,0])
print(kern.eval(x,y))