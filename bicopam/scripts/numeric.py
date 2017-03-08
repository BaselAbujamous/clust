import functools
import math
import numpy as np
import sklearn.metrics.pairwise as skdists


def factors_primesunique(n):
    return [i for i in primes(math.ceil(math.sqrt(n))) if n % i == 0]


def factors_primes(n):
    res = factors_primesunique(n)
    n /= np.prod(res)
    while n > 1:
        localres = factors_primesunique(n)
        res += localres
        n /= np.prod(localres)
    return np.sort(res).tolist()


def factors_all(n):
    return np.unique(functools.reduce(list.__add__,
                                         ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0))).tolist()


def is_prime(n):
    for j in range(2, math.ceil(math.sqrt(n)) + 1):
        if (n % j) == 0:
            return False
    return True


def primes(n):
    if n <= 1:
        return []
    elif n == 2 or n == 3:
        return list(range(2, n + 1))
    else:
        res = [2, 3]
        for i in range(5, n + 1, 2):
            if is_prime(i):
                res.append(i)
    return res


def closest_to_square_factors(n):
    f = factors_all(n)
    sqrtn = math.sqrt(n)
    min_ind = np.argmin([abs(x-sqrtn) for x in f])
    res = [f[min_ind], int(n/f[min_ind])]
    if res[0] > res[1]:
        tmp = res[0]
        res[0] = res[1]
        res[1] = tmp
    return res


def getcondvects(n):
    g = 1
    n2 = 2 ** n
    condvects = np.zeros((n2, n), dtype=bool)
    for m in range(1,n+1):
        m2 = 2 ** m
        m3 = int(m2 / 2)
        n3 = n - m
        for g in range(g, n2, m2):
            for k in range(m3):
                condvects[g + k, n3] = 1
        g = m2
    return condvects


def addaxis(X, V, axis=0):
    Xloc = np.array(X)
    if axis == 0:
        return np.add(Xloc, V)
    if axis == 1:
        return np.add(Xloc.transpose(), V).transpose()
    raise ValueError('Invalid axis value; it has to be 0 or 1')


def subtractaxis(X, V, axis=0):
    Xloc = np.array(X)
    if axis == 0:
        return np.subtract(Xloc, V)
    if axis == 1:
        return np.subtract(Xloc.transpose(), V).transpose()
    raise ValueError('Invalid axis value; it has to be 0 or 1')


def multiplyaxis(X, V, axis=0):
    Xloc = np.array(X)
    if axis == 0:
        return np.multiply(Xloc, V)
    if axis == 1:
        return np.multiply(Xloc.transpose(), V).transpose()
    raise ValueError('Invalid axis value; it has to be 0 or 1')


def divideaxis(X, V, axis=0):
    Xloc = np.array(X)
    np.seterr(divide='ignore', invalid='ignore')
    if axis == 0:
        return np.divide(Xloc, V)
    if axis == 1:
        return np.divide(Xloc.transpose(), V).transpose()
    raise ValueError('Invalid axis value; it has to be 0 or 1')


def isequaltoaxis(X, V, axis=0):
    Xloc = np.array(X)
    [N,M] = Xloc.shape
    if axis == 0:
        if len(V) != M:
            raise ValueError('The length of the vector V should be equal to the number of columns in the matrix X')
        Vloc = np.tile(V, [N, 1])
        return Xloc == Vloc
    if axis == 1:
        if len(V) != N:
            raise ValueError('The length of the vector V should be equal to the number of rows in the matrix X')
        Vloc = np.tile(V, [M, 1]).transpose()
        return Xloc == Vloc
    raise ValueError('Invalid axis value; it has to be 0 or 1')


def all_perms(elements):
    def local_all_perms(elements):
        if len(elements) <=1:
            yield elements
        else:
            for perm in all_perms(elements[1:]):
                for i in range(len(elements)):
                    # nb elements[0:1] works in both string and list contexts
                    yield perm[:i] + elements[0:1] + perm[i:]
    return list(local_all_perms(elements))


def dist_matrices(X1, X2, criterion='euclidean'):
    X1loc = np.array(X1)
    X2loc = np.array(X2)

    if len(X1loc.shape) == 1:
        if len(X2loc.shape) == 1:
            if X1loc.shape[0] == X2loc.shape[0]:
                # As row vectors
                X1loc = X1loc.reshape(1, -1)
                X2loc = X2loc.reshape(1, -1)
            else:
                # As column vectors
                X1loc = X1loc.reshape(-1, 1)
                X2loc = X2loc.reshape(-1, 1)
        else:
            if X1loc.shape[0] == X2loc.shape[1]:
                # Row vector VS. Many rows
                X1loc = X1loc.reshape(1, -1)
            elif X2loc.shape[1] == 1:
                # Column vector VS. Column vector
                X1loc = X1loc.reshape(-1, 1)
            elif X1loc.shape[0] == X2loc.shape[0]:
                # Row vector VS. transposed columns
                X1loc = X1loc.reshape(1, -1)
                X2loc = X2loc.transpose()
            else:
                raise ValueError('Invalid dimensions of X1 and X2')
    elif len(X2loc.shape) == 1:
        if X2loc.shape[0] == X1loc.shape[1]:
            # Many rows VS. row vector
            X2loc = X2loc.reshape(1, -1)
        else:
            raise ValueError('Invalid dimensions of X1 and X2')

    if criterion == 'euclidean':
        return skdists.euclidean_distances(X1loc, X2loc)
    elif criterion == 'hamming':
        raise NotImplementedError('Hamming distance between rows of matrices has not been implemented yet.')
    else:
        raise ValueError('Invalid distance criterion')