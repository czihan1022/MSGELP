import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist


def constructS(fea, k, t):
    nSmp = fea.shape[0]
    bSelfConnected = False

    # Compute the Euclidean distance
    D = cdist(fea, fea, 'sqeuclidean')

    # Initialize the adjacency matrix G
    G = np.zeros((nSmp * (k + 1), 3))

    idNow = 0
    for start in range(0, nSmp, k + 1):
        end = min(start + k + 1, nSmp)
        smpIdx = np.arange(start, end)
        # 当前批次的点与所有点之间的距离
        dist = D[smpIdx, :]
        # 最近的k+1个点的索引 和距离
        idx = np.argsort(dist, axis=1)[:, :k + 1]
        dump = np.take_along_axis(dist, idx, axis=1)
        # 计算热核函数权重，距离越近，权重越大
        dump = np.exp(-dump / (2 * t ** 2))
        # 当前批次的样本数以及填充位置
        nSmpNow = len(smpIdx) * (k + 1)
        G[idNow:idNow + nSmpNow, 0] = np.repeat(smpIdx, k + 1)
        G[idNow:idNow + nSmpNow, 1] = idx.flatten()
        G[idNow:idNow + nSmpNow, 2] = dump.flatten()
        idNow += nSmpNow

    W = csr_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(nSmp, nSmp))

    if not bSelfConnected:
        W.setdiag(0)

    W = W.maximum(W.transpose())

    return W
