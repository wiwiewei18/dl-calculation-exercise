import numpy as np


def relu(x):
    """Applies the ReLU activation function."""
    return np.maximum(0, x)
