import collections
import numpy as np
from copy import deepcopy


def length(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return len(x)
    elif x is None:
        return 0
    else:
        return 1


def numel(x):
    y = np.array(x)
    try:
        size = 1
        for dim in np.shape(y):
            size *= dim
    except:
        size = 1
    return size


def mat2vec(X, dim=1):
    if dim == 1:
        return np.reshape(X, [numel(X), 1])
    elif dim == 2:
        return np.reshape(X, [1, numel(X)])
    else:
        raise ValueError('dim should be either (1) for a column vector or (2) for a row vector')


def matlablike_index2D(X, I0, I1):
    """
    :param X: 2D array or 2D numpy.ndarray
    :param I0: row indices (integers or boolean)
    :param I1: column indices (integers or boolean)
    :return: numpy.ndarray of indexed values
    """
    Y = np.array(X)
    if isinstance(I0, basestring) and I0 in ['all', 'a']:
        I0 = [n for n in range(Y.shape[0])]
    if isinstance(I1, basestring) and I1 in ['all', 'a']:
        I1 = [n for n in range(Y.shape[1])]
    if numel(I0) > 1 and all(isinstance(n, bool) for n in I0):
        Y = Y[np.nonzero(I0)[0]]
    else:
        Y = Y[I0]
    if numel(I1) > 1 and all(isinstance(n, bool) for n in I1):
        Y = Y[:, np.nonzero(I1)[0]]
    else:
        Y = Y[:, I1]
    return Y


def matlablike_indexND(X, indices):
    Y = np.array(X)
    for i in range(len(indices)):
        if type(indices[i]).__module__ == np.__name__ and indices[i].dtype.type == np.bool_:
            Y = Y.take(np.nonzero(indices[i])[0], i)
        elif numel(indices[i]) > 1 and all(isinstance(n,bool) for n in indices[i]):
            Y = Y.take(np.nonzero(indices[i])[0], i)
        else:
            Y = Y.take(indices[i], i)
    return Y


def resolveargumentpairs(refparamsnames, defaults, params=()):
    res = ()
    p = params[0::2]
    v = params[1::2]
    for refp in refparamsnames:
        if refp in p:
            res = res.__add__((v[p.index(refp)],))
        else:
            res = res.__add__((defaults[refparamsnames.index(refp)],))
    return res


def listofarrays2arrayofarrays(X):
    # Check if 2D
    is2d = True
    N2 = -1
    for x in X:
        if isinstance(x, list):
            if N2 == -1:
                N2 = len(x)
            elif len(x) != N2:
                    is2d = False
                    break
        else:
            is2d = False
            break

    #Calculate
    L = len(X)
    if is2d:
        Xout = np.zeros([L, N2], dtype=object)
        for i in range(L):
            for j in range(N2):
                Xout[i,j] = deepcopy(X[i][j])
    else:
        Xout = np.zeros(L, dtype=object)
        for i in range(L):
            Xout[i] = deepcopy(X[i])

    #Return
    return Xout


def maxDepthOfArray(X):
    Xloc = deepcopy(X)
    if isinstance(Xloc, np.ndarray) and Xloc.dtype not in [object, np.ndarray, list, tuple]:
        return len(Xloc.shape)
    elif isinstance(Xloc, (list, tuple, np.ndarray)):
        return max([maxDepthOfArray(x) for x in Xloc]) + 1
    else:
        return 0


def reduceToArrayOfNDArraysAsObjects(X, N):
    Xloc = listofarrays2arrayofarrays(X)
    Xnew = np.array([], dtype=object)
    if maxDepthOfArray(Xloc) == N + 1:
        Xnew = deepcopy(Xloc)
    elif maxDepthOfArray(Xloc) < N + 1:
        Xnew = deepcopy(Xloc)
        for i in range(N + 1 - maxDepthOfArray(Xloc)):
            Xnew = np.expand_dims(Xnew, axis=0)
    else:
        for x in Xloc:
            N0 = maxDepthOfArray(x)
            if N0 == N:
                Xnew = np.concatenate((Xnew, listofarrays2arrayofarrays([Xloc])))
            elif N0 < N:
                Xtmp = x
                for i in range(N - N0):
                    Xtmp = np.expand_dims(Xtmp, axis=0)
                Xnew = np.concatenate((Xnew, Xtmp))
            else:
                Xnew = np.concatenate((Xnew, reduceToArrayOfNDArraysAsObjects(x, N)))
    return Xnew


def flattenAList(l):
    return [item for sublist in l for item in sublist]


def concatenateStrings(listofstrings, delim=', '):
    if isinstance(listofstrings, basestring):
        return listofstrings + ''
    elif len(listofstrings) == 1:
        return listofstrings[0] + ''
    else:
        res = listofstrings[0]
        for s in listofstrings[1:]:
            res += delim + s
        return res


def findArrayInAnotherArray1D(x, y):
    """
    :param x:
    :param y:
    :return: Array with a length similar to x showing x's elements' first occurance indices in y.
    -1 is given for unfound elements.
    """
    index = np.argsort(y)
    sorted_y = y[index]
    sorted_index = np.searchsorted(sorted_y, x)

    xindex = np.take(index, sorted_index, mode="clip")
    mask = y[xindex] != x

    return np.array(np.ma.array(xindex, mask=mask).tolist(fill_value=-1))


def findArrayInSubArraysOfAnotherArray1D(x, y):
    x = np.array(x)
    y = np.array(y)
    y_unfolded = np.array([], dtype=y.dtype)
    y_indices_of_y_unfolded = np.array([], dtype=int)
    for yi in range(len(y)):
        y_unfolded = np.append(y_unfolded, y[yi], axis=0)
        y_indices_of_y_unfolded = np.append(y_indices_of_y_unfolded, [yi for i in range(len(y[yi]))], axis=0)

    I = findArrayInAnotherArray1D(x, y_unfolded)
    I[I > -1] = y_indices_of_y_unfolded[I[I > -1]]
    return I


