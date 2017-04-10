import collections
import math

import numpy as np
import scipy as sp
import sklearn.mixture as skmix

import datastructures as ds
import io
import numeric as nu
import preprocess_data as pp

from joblib import Parallel, delayed
import warnings
import gc


def mseclustersfuzzy(X, B, donormalise=True, GDM=None):
    Xloc = np.array(X)
    Bloc = np.array(B)

    if ds.maxDepthOfArray(Xloc) == 2:
        Xloc = np.expand_dims(Xloc, axis=0)
    Nx = len(Xloc)  # Number of datasets
    if len(Bloc.shape) == 1:
        Bloc = Bloc.reshape(-1, 1)
    M = Bloc.shape[0]  # Number of genes
    K = Bloc.shape[1]  # Number of clusters

    if GDM is None:
        GDMloc = np.ones([Bloc.shape[0], Nx], dtype=bool)
    else:
        GDMloc = np.array(GDM)

    # I commented these two lines after adding GDM
    #if any([True if x.shape[0] != M else False for x in Xloc]):
    #    raise ValueError('Unequal number of genes in datasets and partitions')

    mseC = np.zeros([Nx, K], dtype=float)

    Nk = [np.sum(b) for b in Bloc.transpose()]  # Number of genes per cluster
    Nd = [x.shape[1] for x in Xloc]  # Number of dimensions per dataset

    # Normalise if needed
    if donormalise:
        Xloc = [pp.normaliseSampleFeatureMat(x, 4) for x in Xloc]

    # Calculations
    for nx in range(Nx):
        for k in range(K):
            if Nk[k] == 0:
                mseC[nx, k] = float('nan')
            else:
                Cmeanloc = nu.multiplyaxis(Xloc[nx], Bloc[GDMloc[:, nx], k], axis=1) / Nk[k]  # Weighted mean for the cluster
                tmp = nu.subtractaxis(Xloc[nx], Cmeanloc, axis=0)  # Errors
                tmp = nu.multiplyaxis(tmp, Bloc[GDMloc[:, nx], k], axis=1)  # Weighted errors
                tmp = np.sum(np.power(tmp, 2))  # Squared weighted errors
                mseC[nx, k] = tmp / Nd[nx] / Nk[k]  # Weighted MSE

    return np.mean(mseC, axis=0)


def mseclusters(X, B, donormalise=True, GDM=None):
    Xloc = np.array(X)
    Bloc = np.array(B)

    if ds.maxDepthOfArray(Xloc) == 2:
        Xloc = np.expand_dims(Xloc, axis=0)
    Nx = len(Xloc) # Number of datasets
    if len(Bloc.shape) == 1:
        Bloc = Bloc.reshape(-1, 1)
    M = Bloc.shape[0] # Number of genes
    K = Bloc.shape[1] # Number of clusters

    if GDM is None:
        GDMloc = np.ones([Bloc.shape[0], Nx], dtype=bool)
    else:
        GDMloc = np.array(GDM)

    # I commented these two lines after adding GDM
    #if any([True if x.shape[0] != M else False for x in Xloc]):
    #    raise ValueError('Unequal number of genes in datasets and partitions')

    mseC = np.zeros([Nx, K], dtype=float)

    Nk = [np.sum(b) for b in Bloc.transpose()] # Number of genes per cluster
    Nd = [x.shape[1] for x in Xloc] # Number of dimensions per dataset

    # Normalise if needed
    if donormalise:
        Xloc = [pp.normaliseSampleFeatureMat(x,4) for x in Xloc]

    # Calculations
    for nx in range(Nx):
        for k in range(K):
            if not any(Bloc[:,k]):
                mseC[nx,k] = float('nan')
            else:
                Xlocloc = Xloc[nx][Bloc[GDMloc[:, nx], k], :]
                tmp = nu.subtractaxis(Xlocloc, np.mean(Xlocloc, axis=0), axis=0)
                tmp = np.sum(np.power(tmp,2))
                mseC[nx,k] = tmp / Nd[nx] / Nk[k]

    #io.updateparallelprogress(Nx * K)

    return np.mean(mseC, axis=0)


def mnplotsgreedy(X, B, type='A', params=None, allMSE=None, tightnessweight=1, setsP=None, setsN=None, Xtype='data',
                      mseCache=None, wsets=None, GDM=None, msesummary='average', percentageOfClustersKept=100,
                      smallestClusterSize=11, Xnames=None, ncores=1):
    Xloc = ds.listofarrays2arrayofarrays(X)
    Bloc = ds.reduceToArrayOfNDArraysAsObjects(B, 2)
    L = Xloc.shape[0]  # Number of datasets

    # Fix parameters
    if params is None: params = {}
    if setsP is None: setsP = [x for x in range(int(math.floor(L / 2)))]
    if setsN is None: setsN = [x for x in range(int(math.floor(L / 2)), L)]
    setsPN = np.array(np.concatenate((setsP, setsN), axis=0), dtype=int)
    Xloc = Xloc[setsPN]
    L = Xloc.shape[0]
    if wsets is None:
        wsets = np.array([1 for x in range(L)])
    if GDM is None:
        Ng = np.shape(Xloc[0])[0]
        GDMloc = np.ones([Ng, L], dtype='bool')
    else:
        Ng = np.shape(GDM)[0]
        GDMloc = GDM[:, setsPN]
    if Xnames is None:
        Xnames = ['X{0}'.format(l) for l in range(L)]

    # Put all clusters in one matrix
    N = Bloc.shape[0]  # Number of partitions
    K = [Bloc[i].shape[1] for i in range(N)]  # Number of clusters in each partition

    # One big matrix for all clusters
    BB = Bloc[0]
    for n in range(1, N):
        BB = np.append(BB, Bloc[n], axis=1)
    VMc = np.sum(BB, axis=0)
    NN = len(VMc)  # Total number of clusters

    # Fill Vmse if not provided
    if mseCache is None and allMSE is None:
        # Cache all mse values
        mseCache = np.zeros([NN, L])
        io.resetparallelprogress(NN * L)
        for l in range(L):
            if Xtype == 'files':
                # load files here
                raise NotImplementedError('Xtype "files" has not been implemented yet.')
            elif Xtype == 'data':
                Xtmp = Xloc[l]
            else:
                raise ValueError('Xtype has to be "files" or "data". The given Xtype is invalid.')

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mseCachetmp = Parallel(n_jobs=ncores)\
                    (delayed(mseclusters)
                     (Xtmp, ds.matlablike_index2D(BB, GDMloc[:, l], nn), 0) for nn in range(NN))
                mseCachetmp = [mm[0] for mm in mseCachetmp]
                for nn in range(NN):
                    mseCache[nn, l] = mseCachetmp[nn]

                gc.collect()

                io.updateparallelprogress(NN)

            '''
            for nn in range(NN):
                mseCache[nn, l] = mseclusters(Xtmp, ds.matlablike_index2D(BB, GDMloc[:, l], nn), 0)[0]
            io.log('Done cluster evaluation for {0} have been calculated.'.format(Xnames[l]))
            '''

    # Calculate allMSE if needed (Nx1)
    if allMSE is None:
        if type == 'A':
            wsetsloc = wsets[setsPN]
            wsetsloc = [float(n)/sum(wsetsloc) for n in wsetsloc]
            if msesummary == 'average' or msesummary == 'mean':
                allMSE = np.dot(mseCache[:, setsPN], wsets)
            elif msesummary == 'worse' or msesummary == 'max':
                allMSE = np.max(np.multiply(mseCache[:, setsPN], wsets), axis=1)
            else:
                raise ValueError('msesummary value has to be "average", "mean", "worse", or "max".',
                                 ' "average and "mean" behave similarly, and "worse" and "max" behave similarly.')
        elif type == 'B':
            wsetsP = wsets[setsP]
            wsetsP = [n/sum(wsetsP) for n in wsetsP]
            wsetsN = wsets[setsN]
            wsetsN = [n / sum(wsetsN) for n in wsetsN]
            if msesummary == 'average' or msesummary == 'mean':
                allMSE = np.dot(mseCache[:, setsP] , wsetsP) - np.dot(mseCache[:, setsN] , wsetsN)
            elif msesummary == 'worse' or msesummary == 'max':
                allMSE = np.max(np.multiply(mseCache[:, setsP], wsetsP), axis=1) \
                         - np.max(np.multiply(mseCache[:, setsN], wsetsN), axis=1)
            else:
                raise ValueError('msesummary value has to be "average", "mean", "worse", or "max".',
                                 ' "average and "mean" behave similarly, and "worse" and "max" behave similarly.')
        else:
            raise ValueError('Type should be either A or B; given type is invalid.')

    # Find the distances
    maxx = np.max(allMSE[~np.isnan(allMSE)])
    minx = np.min(allMSE[~np.isnan(allMSE)])
    maxy = np.log10(np.max(VMc))
    miny = 0
    with np.errstate(divide='ignore'):
        allVecs = np.concatenate(([(allMSE - minx) / (maxx - minx)],
                                  [(np.log10(VMc) - miny) / (maxy - miny)]), axis=0).transpose()
    allVecs[:, 0] *= tightnessweight
    allDists = np.array([np.sqrt(1.1 + np.power(tightnessweight, 2)) if np.any(np.isnan(n))
                         else sp.spatial.distance.euclidean(n, [0, 1]) for n in allVecs])
    alpha = 0.0001
    tmp, uVdsI = np.unique(allDists, return_index=True)
    while len(uVdsI) != len(allDists):
        for n in range(len(allDists)):
            if n not in uVdsI:
                allDists[n] += alpha * sp.random.normal()
        tmp, uVdsI = np.unique(allDists, return_index=True)

    # Helper function for greedy solution below
    def mngreedy(Bloc, I, Vds, iter=float('inf')):
        Vdsloc = np.array(Vds)
        res = np.array([False for n in Vdsloc])
        if iter == 0 or not any(I):
            return res
        for n in range(len(I)):
            if not I[n]:
                Vdsloc[n] = float('inf')
        p = np.argmin(Vdsloc)
        res[p] = True
        #II = I
        overlaps = np.dot(ds.matlablike_index2D(Bloc, 'all', p).transpose(), Bloc) > 0
        I &= ~overlaps
        return res | mngreedy(Bloc, I, Vdsloc, iter-1)

    # ** Find greedy solution **
    # Sort clusters based on distances (not important, but benefits the output)
    II = np.argsort(allDists)
    allDists = allDists[II]
    BB = ds.matlablike_index2D(BB, 'a', II)
    allVecs = ds.matlablike_index2D(allVecs, II, 'a')
    allMSE = allMSE[II]
    mseCache = ds.matlablike_index2D(mseCache, II, 'a')
    VMc = VMc[II]

    # include the top XX% of the clusters that have at least smallestClusterSize
    Ismall = VMc < smallestClusterSize
    Inans = np.isnan(allDists)
    tmpDists = [np.max(allDists) if Inans[n] | Ismall[n] else allDists[n] for n in range(len(allDists))]
    percentageOfClustersKept *= float(np.sum(~Ismall)) / len(allDists)
    Iincluded = (tmpDists <= np.percentile(tmpDists, percentageOfClustersKept)) & (np.bitwise_not(Ismall))
    I = mngreedy(BB, Iincluded, allDists)
    B_out = ds.matlablike_index2D(BB, 'a', I)

    # Prepare and return the results:
    params = dict(params, **{
        'tightnessweight': tightnessweight,
        'msesummary': msesummary,
        'percentageofclusterskept': percentageOfClustersKept,
        'smallestclustersize': smallestClusterSize
    })

    MNResults = collections.namedtuple('MNResults',
                                       ['B', 'I', 'allVecs', 'allDists', 'allMSE', 'mseCache', 'Ball', 'params'])
    return MNResults(B_out, I, allVecs, allDists, allMSE, mseCache, BB, params)


def mnplotsdistancethreshold(dists, method='bimodal', returnmodel=False):
    distsloc = np.array(dists).reshape(-1, 1)
    if method == 'bimodal':
        GM = skmix.GaussianMixture(n_components=2)
        GM.fit(distsloc)
        if len(dists) == 1:
            labels = [1]
        else:
            labels = GM.predict(distsloc)
            labels = np.nonzero(labels == labels[0])[0]
        if returnmodel:
            return (labels, GM)
        else:
            return labels
    elif method == 'largestgap' or method == 'largest_gap':
        if len(dists) == 1:
            labels = [1]
        else:
            gaps = np.subtract(dists[1:], dists[0:-1])
            wgaps = np.multiply(gaps, np.arange(len(gaps), 0, -1))
            labels = np.arange(0, np.argmax(wgaps)+1)
        return labels
    else:
        raise ValueError('Invalid method submitted to mnplotsdistancethreshold. '
                         'Use either ''bimodal'' or ''largestgap''')
