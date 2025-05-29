import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist


def constructS(fea, k, t):
    """
    Constructs similarity matrix using heat kernel based on Euclidean distances.

    Parameters:
    - fea: 2D numpy array where each row represents a feature vector of a sample.
    - k: Number of nearest neighbors to consider.
    - t: Temperature parameter for the Gaussian kernel.

    Returns:
    - W: A sparse similarity matrix of shape (n_samples, n_samples).
    """

    nSmp = fea.shape[0]
    bSelfConnected = False
    D = cdist(fea, fea, 'sqeuclidean')

    # Initialize the adjacency matrix G to store the indices and weights
    G = np.zeros((nSmp * (k + 1), 3))

    idNow = 0
    for start in range(0, nSmp, k + 1):
        end = min(start + k + 1, nSmp)
        smpIdx = np.arange(start, end)
        dist = D[smpIdx, :]
        idx = np.argsort(dist, axis=1)[:, :k + 1]
        dump = np.take_along_axis(dist, idx, axis=1)
        dump = np.exp(-dump / (2 * t ** 2))
        nSmpNow = len(smpIdx) * (k + 1)
        G[idNow:idNow + nSmpNow, 0] = np.repeat(smpIdx, k + 1)
        G[idNow:idNow + nSmpNow, 1] = idx.flatten()
        G[idNow:idNow + nSmpNow, 2] = dump.flatten()
        idNow += nSmpNow

    # Construct a sparse matrix W from the G array
    W = csr_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(nSmp, nSmp))

    if not bSelfConnected:
        W.setdiag(0)

    # Ensure symmetry
    W = W.maximum(W.transpose())

    return W
