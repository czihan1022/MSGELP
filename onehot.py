import numpy as np


def onehot(Y, num_c):
    """
    Converts a label array Y into a one-hot encoded matrix.

    Parameters:
    - Y (numpy.ndarray): Array of labels, where each label is an integer representing the class (shape: (num_n, ))
    - num_c (int): Total number of classes.

    Returns:
    - Y_onehot (numpy.ndarray): One-hot encoded matrix (shape: (num_n, num_c)).
    """

    num_n = len(Y)
    Y_onehot = np.zeros((num_n, num_c))
    for i in range(num_n):
        j = Y[i]
        # Set the appropriate position to 1 (convert label to 0-based index)
        Y_onehot[i, j - 1] = 1
    return Y_onehot
