import numpy as np
import datastructures as ds
import sklearn.cluster as skcl
import scipy.cluster.hierarchy as sphc
import scipy.spatial.distance as spdist
import sompy
import io
from glob import maxgenesinsetforpdist


kmeans_init = {}  # This is to cache the centres of the k-means and reuse them


# Main function
def clusterdataset(X, K, D, methods=None, datasetID=-1):
    if methods is None: methods = [['k-means'],['SOMs'],['HC','linkage_method','ward']]
    methodsloc = [n if isinstance(n,(list,tuple,np.ndarray)) else [n] for n in methods]
    #io.log('clusterdataset')
    # Clustering loop
    C = len(methodsloc) # Number of methods
    U = [None] * C
    for ms in range(C):
        if methodsloc[ms][0].lower() in ['k-means', 'kmeans']:
            U[ms] = ckmeans(X, K, datasetID, methodsloc[ms][1:])
        elif methodsloc[ms][0].lower() in ['soms', 'soms-bubble']:
            U[ms] = csoms(X, D, methodsloc[ms][1:])
        elif methodsloc[ms][0].lower() in ['hc', 'hierarchical']:
            U[ms] = chc(X, K, methodsloc[ms][1:])

    io.updateparallelprogress(K * C)

    return U


# Clustering functions
def ckmeans(X, K, datasetID=-1, params=()):
    global kmeans_init

    pnames  = [     'init', 'max_iter', 'n_jobs',  'distance']
    #dflts  = ['k-means++',        300,       -1, 'euclidean']
    dflts   = [       'KA',        300,       -1, 'euclidean']
    if isinstance(params, np.ndarray):
        paramsloc = params.tolist()
    else:
        paramsloc = params
    (init, max_iter, n_jobs, distance) = ds.resolveargumentpairs(pnames, dflts, paramsloc)

    if datasetID in kmeans_init:
        init = kmeans_init[datasetID][0:K]
    elif init == 'KA':
        if X.shape[0] <= maxgenesinsetforpdist:
            init = initclusterKA(X, K, distance)
        else:
            init = initclusterKA_memorysaver(X, K, distance)

    C = skcl.KMeans(K, init=init, max_iter=max_iter, n_jobs=n_jobs).fit(X).labels_
    return clustVec2partMat(C, K)


def csoms(X, D, params=()):
    pnames = ['neighbour', 'learning_rate', 'input_length_ratio']
    dflts  = [        0.1,             0.2,                   -1]
    if isinstance(params, np.ndarray):
        paramsloc = params.tolist()
    else:
        paramsloc = params
    (neighbour, learning_rate, input_length_ratio) = ds.resolveargumentpairs(pnames, dflts, paramsloc)

    Xloc = np.array(X)

    K = D[0] * D[1] # Number of clusters
    N = Xloc.shape[0] # Number of genes
    Ndim = Xloc.shape[1] # Number of dimensions in X

    som = sompy.SOM(D, Xloc)
    som.set_parameter(neighbor=neighbour, learning_rate=learning_rate, input_length_ratio=input_length_ratio)

    centres = som.train(N).reshape(K, Ndim)
    dists = [[spdist.euclidean(c, x) for c in centres] for x in Xloc]
    C = [np.argmin(d) for d in dists]
    return clustVec2partMat(C, K)


def chc(X, K, params=()):
    pnames = ['linkage_method',  'distance']
    dflts  = [          'ward', 'euclidean']
    if isinstance(params, np.ndarray):
        paramsloc = params.tolist()
    else:
        paramsloc = params
    (linkage_method, distance) = ds.resolveargumentpairs(pnames, dflts, paramsloc)

    Z = sphc.linkage(X, method=linkage_method, metric=distance)
    C = sphc.fcluster(Z, K, criterion='maxclust')
    return clustVec2partMat(C, K)


# Other related functions
def initclusterKA(X, K, distance='euclidean'):
    M = X.shape[0]
    Dist = spdist.pdist(X, metric=distance)  # MxM (condensed)
    Dist = spdist.squareform(Dist)  # MxM
    ResultInd = [0 for i in range(K)]
    Xmean = np.mean(X, axis=0)  # The mean of all rows of X
    Dmean = spdist.cdist(X, [Xmean])  # Distances between rows of X and the mean of X

    # The first centre is the closest point to the mean:
    ResultInd[0] = np.argmin(Dmean)
    io.updateparallelprogress(K)

    for k in range(K-1):
        D = np.min(Dist[:, ResultInd[0:k+1]], axis=1)  # Mx1
        C = [0 for m in range(M)]  # M points (e.g. genes)  # Mx1
        for m in range(M):
            if m in ResultInd:
                continue
            tmp = D - Dist[:, m]  # Mx1 differences
            tmp[tmp < 0] = 0  # All negatives make them zeros
            C[m] = np.sum(tmp)
        ResultInd[k+1] = np.argmax(C)
        io.updateparallelprogress(K)


    Result = X[ResultInd]  # These points are the selected K initial cluster centres

    return Result


'''This is the same as initclusterKA, but does not calculate the pdist amongst all objects at once.
It rather uses loops and therefore saves the memory.'''
def initclusterKA_memorysaver(X, K, distance='euclidean'):
    M = X.shape[0]
    #Dist = spdist.pdist(X, metric=distance)  # MxM (condensed)
    #Dist = spdist.squareform(Dist)  # MxM
    ResultInd = [0 for i in range(K)]
    Xmean = np.mean(X, axis=0)  # The mean of all rows of X
    Dmean = spdist.cdist(X, [Xmean], metric=distance)  # Distances between rows of X and the mean of X

    # The first centre is the closest point to the mean:
    ResultInd[0] = np.argmin(Dmean)
    io.updateparallelprogress(K)

    for k in range(K-1):
        D = spdist.cdist(X, X[ResultInd[0:k+1]], metric=distance)  # (M)x(k+1) Dists of objects to the selected centres
        D = np.min(D, axis=1)  # Mx1: Distances of each of the M objects to its closest already selected centre
        C = [0 for m in range(M)]  # M objects (e.g. genes)  # Mx1
        for m in range(M):
            if m in ResultInd:
                continue
            Dists_m = spdist.cdist(X, [X[m]])  # Mx1 distances between all M objects and the m_th object
            tmp = D.reshape(M, 1) - Dists_m  # Mx1 differences
            tmp[tmp < 0] = 0  # All negatives make them zeros
            C[m] = np.sum(tmp)
        ResultInd[k+1] = np.argmax(C)
        io.updateparallelprogress(K)

    Result = X[ResultInd]  # These objects are the selected K initial cluster centres

    return Result


def cache_kmeans_init(X, K, methods, datasetID):
    global kmeans_init

    if datasetID == -1:
        return

    # Get the k-means parameters
    methodsloc = [n if isinstance(n, (list, tuple, np.ndarray)) else [n] for n in methods]
    kmeansFound = False
    for ms in range(len(methodsloc)):
        if methodsloc[ms][0].lower() in ['k-means', 'kmeans']:
            params = methodsloc[ms][1:]
            pnames = ['init', 'max_iter', 'n_jobs',  'distance']
            dflts  = [  'KA',        300,       -1, 'euclidean']
            if isinstance(params, np.ndarray):
                paramsloc = params.tolist()
            else:
                paramsloc = params
            (init, max_iter, n_jobs, distance) = ds.resolveargumentpairs(pnames, dflts, paramsloc)
            kmeansFound = True

    # Perform initialisation over the largest K value and cache it, if k-means was found and init is some 'KA'
    if kmeansFound:
        if init == 'KA':
            if X.shape[0] <= maxgenesinsetforpdist:
                kmeans_init[datasetID] = initclusterKA(X, np.max(K), distance)
            else:
                kmeans_init[datasetID] = initclusterKA_memorysaver(X, np.max(K), distance)


def clustVec2partMat(C, K=None):
    if K is None: K = np.max(C)+1  # Number of clusters
    N = len(C)  # Number of genes
    baseInd = np.min(C)  # The first cluster's index (0 or 1)
    Cfixed = C - baseInd

    U = np.reshape([False]*(N*K), [N, K])
    for i in range(N):
        if Cfixed[i] >= 0:
            U[i, Cfixed[i]] = True

    return U


def partMat2clustVec(U):
    return [np.nonzero(u)[0][0] if np.any(u) else -1 for u in U]