import numpy as np
import math
from typing import Tuple, Dict
import os


def _calculate_dists(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    size1 = X1.shape
    size2 = X2.shape

    G = np.sum(X1 * X1, axis=1)
    H = np.sum(X2 * X2, axis=1)

    Q = np.transpose(np.tile(G, (size2[0], 1)))
    R = np.tile(np.transpose(H), (size1[0], 1))

    dists = Q + R - 2 * np.matmul(X1, np.transpose(X2))
    return dists


def _rbf_dot(X1: np.ndarray, X2: np.ndarray, deg: float) -> np.ndarray:
    dists = _calculate_dists(X1, X2)
    H = np.exp(-dists / 2 / (deg * deg))
    return H


def _compute_kernel_size(X1: np.ndarray, X2: np.ndarray, max_samples: int = 100) -> float:
    num_samples = min(max_samples, len(X1) + len(X2))
    Z = np.concatenate((X1, X2[:(num_samples - len(X1)),...]), axis=0) if num_samples > len(X1) else X1[:num_samples]
    dists = _calculate_dists(Z, Z)

    dists = dists - np.tril(dists)
    dists = np.reshape(dists, (-1))
    sig = math.sqrt(0.5 * np.median(dists[dists > 0]))
    return sig


def mmdTestBoot(X: np.ndarray, Y: np.ndarray, alpha: float, params: Dict) -> Tuple[float, float]:
    m = X.shape[0]

    sig = _compute_kernel_size(X, Y) if 'sig' not in params or params['sig'] <= 0 else params['sig']

    K = _rbf_dot(X, X, sig)
    L = _rbf_dot(Y, Y, sig)
    KL = _rbf_dot(X, Y, sig)

    # MMD statistic. Here we use biased
    # v-statistic. NOTE: this is m * MMD_b
    testStat = 1 / m * np.sum(K + L - KL - np.transpose(KL))

    Kz = np.concatenate((np.concatenate((K, KL), axis=1), np.concatenate((np.transpose(KL), L), axis=1)), axis=0)

    num_shuffles = params['shuff']
    threshFileName = f'mmdTestThresh{m}.txt'
    if 'bootForce' in params and params['bootForce'] or not os.path.isfile(threshFileName):
        MMDarr = np.zeros(num_shuffles)
        for whichSh in range(num_shuffles):
            indShuff = np.random.permutation(2*m)
            KzShuff = Kz[:, indShuff][indShuff, :]
            K = KzShuff[:m, :m]
            L = KzShuff[m:(2*m), m:(2*m)]
            KL = KzShuff[:m,m:(2*m)]

            MMDarr[whichSh] = 1 / m * np.sum(K + L - KL - np.transpose(KL))

        #np.savetxt(threshFileName, MMDarr)
    else:
        MMDarr = np.loadtxt(threshFileName)

    thresh = np.quantile(MMDarr, 1 - alpha)

    return testStat, thresh

