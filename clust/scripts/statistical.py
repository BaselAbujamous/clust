import numpy as np
import scipy.stats as sps
from datastructures import length
from copy import deepcopy
import math


def pvalue(N, M, n, m):
    N = deepcopy(N)
    M = deepcopy(M)
    n = deepcopy(n)
    m = deepcopy(m)
    maxlen = max([length(N), length(M), length(n), length(m)])
    if maxlen > 1:
        if length(N) == 1:
            N = [N for i in range(maxlen)]
        elif length(N) != maxlen:
            raise ValueError('Inequally long vectors have been provided to this function')
        if length(M) == 1:
            M = [M for i in range(maxlen)]
        elif length(M) != maxlen:
            raise ValueError('Inequally long vectors have been provided to this function')
        if length(n) == 1:
            n = [n for i in range(maxlen)]
        elif length(n) != maxlen:
            raise ValueError('Inequally long vectors have been provided to this function')
        if length(m) == 1:
            m = [m for i in range(maxlen)]
        elif length(m) != maxlen:
            raise ValueError('Inequally long vectors have been provided to this function')
        return [pvalue(N[i],M[i],n[i],m[i]) for i in range(maxlen)]
    else:
        hg = sps.hypergeom(N, M, n)
        if m > M or m > n:
            m = min(M, n)
        return sum(hg.pmf(np.arange(m, min(M + 1, n + 1))))


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return average, math.sqrt(variance)


def weighted_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return math.sqrt(variance)


def weighted_std_axis(X, weights, axis=0):
    if axis == 0:
        X = np.transpose(X)

    return [weighted_std(x, weights) for x in X]

