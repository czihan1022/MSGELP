import numpy as np


def onehot(Y, num_c):

    num_n = len(Y)
    Y_onehot = np.zeros((num_n, num_c))
    for i in range(num_n):
        j = Y[i]
        Y_onehot[i, j - 1] = 1  # 注意将索引转换为从 0 开始
    return Y_onehot
