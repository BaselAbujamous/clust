import numpy as np
import datastructures as ds
import sklearn.cluster as skcl
import scipy.cluster.hierarchy as sphc
import scipy.spatial.distance as spdist
import sompy
import io


# Main function
def clusterdataset(X, K, D, methods=None):
    if methods is None: methods = [['k-means'],['SOMs'],['HC','linkage_method','ward']]
    methodsloc = [n if isinstance(n,(list,tuple,np.ndarray)) else [n] for n in methods]
    #io.log('clusterdataset')
    # Clustering loop
    C = len(methodsloc) # Number of methods
    U = [None] * C
    for ms in range(C):
        if methodsloc[ms][0].lower() in ['k-means', 'kmeans']:
            U[ms] = ckmeans(X, K, methodsloc[ms][1:])
        elif methodsloc[ms][0].lower() in ['soms', 'soms-bubble']:
            U[ms] = csoms(X, D, methodsloc[ms][1:])
        elif methodsloc[ms][0].lower() in ['hc', 'hierarchical']:
            U[ms] = chc(X, K, methodsloc[ms][1:])

    io.updateparallelprogress(K * C)

    return U


# Clustering functions
def ckmeans(X, K, params=()):
    pnames  = [     'init','max_iter', 'n_jobs']
    dflts   = ['k-means++',       300,       -1]
    (init, max_iter, n_jobs) = ds.resolveargumentpairs(pnames, dflts, params)

    if init == 'KA':
        init = initclusterKA(X, K, inittype=init)

    C = skcl.KMeans(K, init=init, max_iter=max_iter, n_jobs=n_jobs).fit(X).labels_
    return clustVec2partMat(C, K)


def csoms(X, D, params=()):
    pnames = ['neighbour', 'learning_rate', 'input_length_ratio']
    dflts  = [        0.1,             0.2,                   -1]
    (neighbour, learning_rate, input_length_ratio) = ds.resolveargumentpairs(pnames, dflts, params)

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
def initclusterKA(X, K, distance='euclidean', inittype='KA'):
    raise NotImplementedError('Kaufman''s initialisation has not been implemented yet.')


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