import numpy as np


def EProjSimplex_new(v, k=1):
    """
    Projects a vector v onto the simplex defined by the constraint
    that the sum of its elements is equal to k and all elements are
    non-negative.

    Parameters:
    - v: 1D numpy array to be projected onto the simplex.
    - k: The value that the sum of the projected vector should equal (default is 1).

    Returns:
    - x: The projected vector that lies on the simplex.
    """

    ft = 1
    n = len(v)

    # Center the vector v by subtracting its mean and adjusting to ensure the sum equals k
    v0 = v - np.mean(v) + k / n
    vmin = np.min(v0)

    # Check if the minimum value of v0 is negative
    if vmin < 0:
        f = 1
        lambda_m = 0
        while abs(f) > 1e-10:
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = np.sum(posidx)
            g = -npos
            f = np.sum(v1[posidx]) - k
            lambda_m -= f / g
            ft += 1
            if ft > 500:
                x = np.maximum(v1, 0)
                break
        x = np.maximum(v1, 0)
    else:
        x = v0

    return x
