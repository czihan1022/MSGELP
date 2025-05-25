import numpy as np


def L2_distance_1(a, b):
    """
    计算两个矩阵 a 和 b 的平方欧氏距离。

    参数:
    - a: 数据矩阵，每列为一个数据点 (n_features, n_samples_a)
    - b: 数据矩阵，每列为一个数据点 (n_features, n_samples_b)

    返回:
    - d: a 和 b 之间的距离矩阵 (n_samples_a, n_samples_b)
    """
    if a.shape[0] == 1:
        # 如果 a 和 b 只有一个特征，则添加一行 0 以使得其至少有两行
        a = np.vstack([a, np.zeros((1, a.shape[1]))])
        b = np.vstack([b, np.zeros((1, b.shape[1]))])

    # 计算每列的平方和
    aa = np.sum(a ** 2, axis=0)
    bb = np.sum(b ** 2, axis=0)
    ab = np.dot(a.T, b)

    # 计算平方欧氏距离矩阵
    d = np.add.outer(aa, bb) - 2 * ab

    # 确保距离非负
    d = np.real(d)
    d = np.maximum(d, 0)

    return d
