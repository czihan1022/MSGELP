import numpy as np


def EProjSimplex_new(v, k=1):
    # 初始设置
    ft = 1
    n = len(v)

    # 对 v 做均值平移，得到 v0
    v0 = v - np.mean(v) + k / n
    vmin = np.min(v0)

    # 检查 v0 的最小值
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
