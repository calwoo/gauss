import numpy as np


class Kernel:
    """initializes default L2 distance kernel"""
    def eval(self, x, y):
        return np.sum((x-y)**2)