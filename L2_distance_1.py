import numpy as np


def L2_distance_1(a, b):
    """
    Computes the squared Euclidean distance between two matrices a and b.

    Parameters:
    - a (numpy.ndarray): Data matrix where each column is a data point (shape: n_features, n_samples_a)
    - b (numpy.ndarray): Data matrix where each column is a data point (shape: n_features, n_samples_b)

    Returns:
    - d (numpy.ndarray): Distance matrix containing squared distances (shape: n_samples_a, n_samples_b)
    """

    if a.shape[0] == 1:
        a = np.vstack([a, np.zeros((1, a.shape[1]))])
        b = np.vstack([b, np.zeros((1, b.shape[1]))])

    aa = np.sum(a ** 2, axis=0)
    bb = np.sum(b ** 2, axis=0)
    ab = np.dot(a.T, b)

    d = np.add.outer(aa, bb) - 2 * ab

    # Ensure the distances are non-negative
    d = np.real(d)
    d = np.maximum(d, 0)

    return d
