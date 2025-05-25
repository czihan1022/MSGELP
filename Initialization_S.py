import numpy as np
from constructS import constructS

def initialization_s(Xs, Xt):
    # 将源域和目标域数据合并
    X = np.hstack((Xs, Xt))

    # 构建基于 KNN 的邻接矩阵
    k = 5
    t = 1
    S = constructS(X.T, k, t)

    # 稀疏矩阵转为稠密矩阵
    S = S.toarray()
    return S
