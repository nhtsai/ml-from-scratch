"""Utils"""

import numpy as np

def min_max_normalize(X):
    maximum = np.max(X)
    minimum = np.min(X)
    return (X - minimum) / (maximum - minimum)


def squared_error(y_pred, y_test):
    return np.sum(np.square(y_pred - y_test))