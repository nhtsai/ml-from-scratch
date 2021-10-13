"""Utils"""

import numpy as np

def min_max_normalize(X):
    maximum = np.max(X)
    minimum = np.min(X)
    return (X - minimum) / (maximum - minimum)