import numpy as np
from constructS import constructS

def initialization_s(Xs, Xt):
    """
    Initializes the adjacency matrix for both source and target domain data.

    Parameters:
    Xs (numpy.ndarray): Source domain data, shape (m, n_s).
    Xt (numpy.ndarray): Target domain data, shape (m, n_t).

    Returns:
    numpy.ndarray: A dense adjacency matrix combining both source and target domains.
    """

    X = np.hstack((Xs, Xt))
    k = 5
    t = 1
    S = constructS(X.T, k, t)

    # Convert sparse matrix to dense matrix
    S = S.toarray()
    return S
