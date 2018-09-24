import numpy as np
import scipy.stats as sps
from datastructures import length
from copy import deepcopy
import scipy.stats as stats
import numeric as nu
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


def mean_distance_metric(X, distance='euclidean'):
    """
    Finds the mean of the rows in X by taking into consideration the given distance metric
    :param X: Matrix of data
    :param distance: Distance metric (default: euclidean)
    :return: A row vector which is the mean of X
    """
    distance = distance.lower()
    if distance in ['euc', 'euclidean', 'city', 'cityblock', 'manhattan', 'maha', 'mahalanobis']:
        return np.mean(X, axis=0)
    elif distance == 'hamming':
        return stats.mode(X, axis=0, nan_policy='omit')[0][0]
    else:
        raise Exception('Unknown distance metric for mean calculations')


def mse_distance_metric(X, distance='euclidean'):
    """
    Finds the MSE of the rows in X by taking into consideration the given distance metric
    :param X: Matrix of data
    :param distance: Distance metric (default: euclidean)
    :return: A single summary MSE value
    """
    distance = distance.lower()
    mean = mean_distance_metric(X, distance)
    Nd = X.shape[1]  # Number of dimensions
    Nk = X.shape[0]  # Number of genes in the cluster

    if distance in ['euc', 'euclidean', 'city', 'cityblock', 'manhattan']:
        tmp = nu.subtractaxis(X, mean, axis=0)
        tmp = np.sum(np.power(tmp, 2))
        return tmp / Nd / Nk

    elif distance == 'hamming':
        tmp = nu.isequaltoaxis(X, mean, axis=0) + 0
        tmp = np.sum(tmp)
        return tmp / Nd / Nk

    elif distance in ['maha', 'mahalanobis']:
        raise Exception('Mahalanobis distance is not fully implemented yet.')
    else:
        raise Exception('Unknown distance metric for mean calculations')
