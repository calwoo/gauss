import numpy as np
import matplotlib.pyplot as plt



def kernel_heatmap(kernel):
    """creates a sample heatmap of the given kernel"""
    heat = np.zeros((25,25))
    xs = np.arange(-5.0, 5.0, step=0.4)
    for i, x in enumerate(xs):
        for j, y in enumerate(xs):
            heat[i][j] = kernel.eval(x, y)
    plt.imshow(heat, cmap="PuRd")
    plt.show()