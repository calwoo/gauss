import numpy as np


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

    