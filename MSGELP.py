import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from onehot import onehot
import scipy
import Initialization_S
from L2_distance_1 import L2_distance_1
from EProjSimplex_new import EProjSimplex_new


def MSGELP(Xs_list, Ys_list, Xs, Ys, Xt, Yt, Ft_init, c, iterations, flag, param_list):
    alpha, beta, lambda_, gamma, p, q = param_list[0], param_list[1], param_list[2], param_list[3], param_list[4], param_list[5]
    Ft = Ft_init
    Ft_list = [Ft_init.copy() for _ in range(len(Xs_list))]
    Sk_list = []

    for xsk in Xs_list:
        S = Initialization_S.initialization_s(xsk, Xt)
        Sk_list.append(S)

    # init A
    d, ns = Xs.shape
    _, nt = Xt.shape
    n = ns + nt
    X = np.hstack((Xs, Xt))
    Ys_onehot = onehot(Ys, c)  # One-hot encoding
    H = np.eye(n) - (1 / n) * np.ones((n, n))
    Ta = Xs @ Ys_onehot / ns - Xt @ Ft / nt
    w, V = scipy.linalg.eig(Ta @ Ta.T + 0.1 * np.eye(d), X @ H @ X.T)  # 求矩阵广义特征值与特征向量 w是特征值矩阵 V是特征向量矩阵
    ind = np.argsort(w)  # 返回的是元素值从小到大排序后的索引值的数组
    A = V[:, ind[:p]]  # 对应的特征向量

    Xs_hat_list = []
    Ys_onehot_list = []
    for i in range(len(Xs_list)):
        Xs_hat_list.append(A.T @ Xs_list[i])
        Ys_onehot_list.append(onehot(Ys_list[i], c))
    Xt_hat = A.T @ Xt

    # init Bk
    B_list = []
    for i in range(len(Xs_list)):
        _, nsk = Xs_list[i].shape
        nk = nsk + nt
        Hk = np.eye(nk) - (1 / nk) * np.ones((nk, nk))
        X_hat = np.hstack((Xs_hat_list[i], Xt_hat))
        Tb = Xs_hat_list[i] @ Ys_onehot_list[i] / nsk - Xt_hat @ Ft_list[i] / nt
        S = Initialization_S.initialization_s(Xs_hat_list[i], Xt_hat)
        D = np.diag(np.sum(S, axis=1))
        L = D - S
        w, V = scipy.linalg.eig(alpha * Tb @ Tb.T + 2 * beta * X_hat @ L @ X_hat.T + 0.1 * np.eye(p), X_hat @ Hk @ X_hat.T)  # 求矩阵广义特征值与特征向量 w是特征值矩阵 V是特征向量矩阵
        ind = np.argsort(w)  # 返回的是元素值从小到大排序后的索引值的数组
        B = V[:, ind[:q]]  # 对应的特征向量
        B_list.append(B)

    predict_label = np.argmax(Ft, axis=1) + 1
    acc = np.sum(predict_label == Yt) / len(Yt)
    # print(acc)

    F_list = []
    Tb_list = []
    precision = 0
    recall = 0
    f1 = 0
    roc = 0
    for i in range(iterations):
        for j in range(len(Xs_list)):
            F = np.vstack((Ys_onehot_list[j], Ft_list[j]))
            if i == 0:
                F_list.append(F)
            else:
                F_list[j] = F
            X_hat = np.hstack((Xs_hat_list[j], Xt_hat))
            d_bx_k = L2_distance_1(B_list[j].T @ X_hat, B_list[j].T @ X_hat)
            d_f_k = L2_distance_1(F.T, F.T)
            dk = beta * d_bx_k + 0.5 * gamma * d_f_k

            _, nsk = Xs_list[j].shape
            nk = nsk + nt

            # Update Sk
            for t in range(nk):
                m = -dk[t, :] / (2 * lambda_)
                Sk_list[j][:, t] = EProjSimplex_new(m)

            Sk_list[j] = (Sk_list[j] + Sk_list[j].T) / 2

            # Laplacian matrix
            D = np.diag(np.sum(Sk_list[j], axis=1))
            L = D - Sk_list[j]
            Lst = L[:nsk, nsk:]
            Dtt = D[nsk:, nsk:]
            Stt = Sk_list[j][nsk:, nsk:]

            # Update Ftk
            Z = Xt_hat.T @ (B_list[j] @ B_list[j].T) @ Xt_hat
            M = gamma / nsk * Ys_onehot_list[j].T @ Lst - alpha / nt * Ys_onehot_list[j].T / nsk @ Xs_hat_list[j].T @ (B_list[j] @ B_list[j].T) @ Xt_hat

            for t in range(nt):
                bb = np.zeros(c)
                for u in range(nt):
                    ft = Ft_list[j][u, :].reshape(-1)
                    bb += 0.5 * gamma * Stt[t, u] * ft - alpha / (2 * nt * nt) * Z[t, u] * ft

                b = bb - M[:, t]
                mm = b / (gamma * Dtt[t, t])
                v = EProjSimplex_new(mm)
                Ft_list[j][t, :] = v

        # Updete Ft
        Ft = np.mean(Ft_list, axis=0)
        predict_label = np.argmax(Ft, axis=1) + 1

        acc = np.sum(predict_label == Yt) / len(Yt)
        precision = precision_score(Yt, predict_label, average='macro')
        recall = recall_score(Yt, predict_label, average='macro')
        f1 = f1_score(Yt, predict_label, average='macro')

        Yt_binarized = label_binarize(Yt, classes=[1, 2, 3, 4])
        roc = roc_auc_score(Yt_binarized, Ft, multi_class='ovr', average='macro')


        # Update A
        Ta = Xs @ Ys_onehot / ns - Xt @ Ft / nt
        w, V = scipy.linalg.eig(Ta @ Ta.T + 0.1 * np.eye(d), X @ H @ X.T)
        ind = np.argsort(w)
        A = V[:, ind[:p]]

        for j in range(len(Xs_hat_list)):
            Xs_hat_list[j] = A.T @ Xs_list[j]
        Xt_hat = A.T @ Xt

        for j in range(len(Xs_list)):
            _, nsk = Xs_list[j].shape
            nk = nsk + nt
            D = np.diag(np.sum(Sk_list[j], axis=1))
            L = D - Sk_list[j]

            # Update Bk
            X_hat = np.hstack((Xs_hat_list[j], Xt_hat))
            Tb = Xs_hat_list[j] @ Ys_onehot_list[j] / nsk - Xt_hat @ Ft_list[j] / nt
            if i == 0:
                Tb_list.append(Tb)
            else:
                Tb_list[j] = Tb
            Hk = np.eye(nk) - (1 / nk) * np.ones((nk, nk))
            w, V = scipy.linalg.eig(alpha * Tb @ Tb.T + 2 * beta * X_hat @ L @ X_hat.T + 0.1 * np.eye(p), X_hat @ Hk @ X_hat.T)  # 求矩阵广义特征值与特征向量 w是特征值矩阵 V是特征向量矩阵
            ind = np.argsort(w)  # 返回的是元素值从小到大排序后的索引值的数组
            B = V[:, ind[:q]]  # 对应的特征向量
            B_list[j] = B

    if flag == 1:
        print('best:', acc)
        print('precision:', precision)
        print('recall:', recall)
        print('f1:', f1)
        print('roc:', roc)

    return acc
