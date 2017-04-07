import collections
import math
import sys

import numpy as np

import clustering as cl
import datastructures as ds
import io
import mnplots as mn
import numeric as nu

from joblib import Parallel, delayed
import joblib
import warnings
import gc


def binarise(U, technique, param=0.0):
    K = np.shape(U)[1]
    allZerosInd = np.sum(U, axis=1) == 0
    technique = technique.lower()
    if technique in ['union', 'ub']:
        B = U > 0
    elif technique in ['intersection', 'ib']:
        B = U == 1
    elif technique in ['max', 'mvb']:
        B = nu.isequaltoaxis(U, np.max(U, axis=1), axis=1)
    elif technique in ['valuethreshold', 'value', 'vtb']:
        B = U >= param
    elif technique in ['stdthresh', 'std']:
        B = (nu.isequaltoaxis(U, np.max(U, axis=1), axis=1)) & \
            (np.tile(np.std(U, axis=1), [K, 1]).transpose() > param)
    elif technique in ['difference', 'diff', 'dtb']:
        Usorted = np.sort(U, axis=1)
        diff = Usorted[:,-1] - Usorted[:,-2]
        B = (nu.isequaltoaxis(U, np.max(U, axis=1), axis=1)) & \
            (np.tile(diff, [K, 1]).transpose() > param)
    elif technique in ['top', 'tb']:
        B = nu.subtractaxis(U, np.max(U, axis=1), axis=1) <= param
    else:
        raise ValueError('The given technique is invalid.')
    B[allZerosInd] = 0
    return np.array(B, dtype='bool')


def fuzzystretch(X, x0=None):
    Xloc = np.array(X)
    if x0 is None:
        x0 = np.array([np.mean(xrow[xrow > 0]) for xrow in Xloc])
        x0[x0 == 1] = 0.5
    elif ds.numel(x0) == 1:
        x0 = np.array([x0 for i in range(Xloc.shape[0])])
    elif ds.numel(x0) != Xloc.shape[0]:
        raise ValueError('The parameter x0 should either be a single value or a vector of length equal to the number '
                         'of rows in X. It can also be left ungiven as it has a default value.')

    y = np.zeros(Xloc.shape)
    for i in range(Xloc.shape[0]):
        xrow = Xloc[i]
        xt = xrow
        xt[xrow < x0[i]] = (np.pi * xrow[xrow < x0[i]]) / (2*x0[i]) - np.pi / 2
        xt[xrow >= x0[i]] = (xrow[xrow >= x0[i]] - x0[i]) * np.pi / (2*(1-x0[i]))

        yt = np.zeros(len(xt))
        yt[xrow < x0[i]] = x0[i] + x0[i] * np.sin(xt[xrow < x0[i]])
        yt[xrow >= x0[i]] = x0[i] + (1-x0[i]) * np.sin(xt[xrow >= x0[i]])

        y[i] = yt

    return y


def clustDist(B1, B2, X=None, criterion='direct_euc'):
    if criterion == 'direct_euc':
        D = nu.dist_matrices(B1.transpose(), B2.transpose())
    elif criterion == 'centres_euc':
        centres1 = nu.divideaxis(np.dot(B1.transpose(), X), np.sum(B1, axis=0), axis=1)
        centres2 = nu.divideaxis(np.dot(B2.transpose(), X), np.sum(B2, axis=0), axis=1)

        D = nu.dist_matrices(centres1, centres2)
        if np.any(~np.isnan(D)):
            m = np.max(D[~np.isnan(D)])
            D[np.isnan(D)] = m + 1
        else:
            D = np.ones(D.shape)
    elif criterion == 'union_std':
        K1 = B1.shape[1]
        K2 = B2.shape[1]
        D = np.zeros([K1, K2])

        for k1 in range(K1):
            for k2 in range(K2):
                bUnion = np.max([B1[:,k1], B2[:,k2]], axis = 0)  # (1)x(Ng)
                bCentre = np.dot(bUnion, X) / np.sum(bUnion)  # (1)x(Xdim)
                distsFromCentre = nu.dist_matrices(X, bCentre)  # (Ng)x(1)
                D[k1, k2] = np.dot(bUnion, distsFromCentre) / np.sum(bUnion)  # (1)x(1)
    elif criterion == 'hamming':
        D = nu.dist_matrices(B1.transpose(), B2.transpose(), criterion='hamming')
    else:
        raise ValueError('Invalid distance criterion provided.')

    return D


def relabelClusts(Ref, Bin, method='minmin', X=None, distCriterion='direct_euc'):
    # Fix parameters
    Refloc = np.array(Ref)
    Binloc = np.array(Bin)
    if sum(np.array(Refloc.shape) == 1):
        Refloc = cl.clustVec2partMat(Refloc)
    if sum(np.array(Binloc.shape) == 1):
        Binloc = cl.clustVec2partMat(Binloc)
    Kin = Binloc.shape[1]

    # Helping functions:
    def relabel_brute(Ref, Bin, X=None, distCriterion='direct_euc'):
        Kref = Ref.shape[1]
        Kin = Bin.shape[1]

        D = clustDist(Ref, Bin, X, distCriterion)

        Permutations = np.array(nu.all_perms([i for i in range(Kin)]))
        PermsSums = np.zeros(len(Permutations))

        for l1 in range(len(Permutations)):
            for l2 in range(min(Kin,Kref)):
                PermsSums[l1] += D[l2, Permutations[l1, l2]] # D(ii, Permutations(ll, ii))

        best = np.argmin(PermsSums)

        return Permutations[best]

    def relabel_minmin(Ref, Bin, X=None, distCriterion='direct_euc'):
        Kref = Ref.shape[1]
        Kin = Bin.shape[1]

        D = clustDist(Ref, Bin, X, distCriterion)

        Perm = np.zeros(Kin) - 1

        maxval = np.max(D) + 1

        for l1 in range(min(Kin, Kref)):
            mi = np.min(D, axis=0)
            mm = np.min(mi)
            col = np.where(mi == mm)[0][0]
            row = np.where(D[:,col] == mm)[0][0]

            Perm[row] = col

            D[row] = maxval
            D[:, col] = maxval

        if Kin > Kref:
            Perm[Perm == -1] = np.setdiff1d(range(Kin), Perm)

        return np.array(Perm, dtype=int)

    def relabel_minmax(Ref, Bin, X=None, distCriterion='direct_euc'):
        Kref = Ref.shape[1]
        Kin = Bin.shape[1]

        D = clustDist(Ref, Bin, X, distCriterion)

        Perm = np.zeros(Kin) - 1

        maxval = np.max(D) + 1
        minval = -1

        for l1 in range(min(Kin, Kref)):
            mi = np.min(D, axis=0)
            mm = np.max(mi)
            col = np.where(mi == mm)[0][0]
            row = np.where(D[:,col] == mm)[0][0]

            Perm[row] = col

            D[row] = maxval
            D[:, col] = minval

        if Kin > Kref:
            Perm[Perm == -1] = np.setdiff1d(range(Kin), Perm)

        return np.array(Perm, dtype=int)

    # Invoke the relevent relabelling function
    if method == 'brute':
        perm = relabel_brute(Refloc, Binloc, X, distCriterion)
    elif method == 'minmin_strict':
        perm = relabel_minmin(Refloc, Binloc, X, distCriterion)
    elif method == 'minmin':
        if Kin < 8:
            perm = relabel_brute(Refloc, Binloc, X, distCriterion)
        else:
            perm = relabel_minmin(Refloc, Binloc, X, distCriterion)
    elif method == 'minmax_strict':
        perm = relabel_minmax(Refloc, Binloc, X, distCriterion)
    elif method == 'minmax':
        if Kin < 8:
            perm = relabel_brute(Refloc, Binloc, X, distCriterion)
        else:
            perm = relabel_minmax(Refloc, Binloc, X, distCriterion)

    return Binloc[:, perm]


def generateCoPaM(U, relabel_technique='minmin', w=None, X=None, distCriterion='direct_euc', K=0, GDM=None):
    # Helping functions
    def calwmeans(w):
        wm = [np.mean(calwmeans(ww)) if isinstance(ww, (list, tuple, np.ndarray)) else np.mean(ww) for ww in w]
        return np.array(wm)
    def CoPaMsdist(CoPaM1, CoPaM2):
        return np.linalg.norm(CoPaM1 - CoPaM2)
    def orderpartitions(U, method='rand', X=None, GDM=None):
        if method == 'rand':
            return np.random.permutation(range(len(U))), None
        elif method == 'mn':
            # TODO: Implement ranking partitions based on M-N plots
            raise NotImplementedError('Ranking partitions based on the M-N plots logic has not been implemented yet.')
        elif method == 'mse':
            R = len(U)
            mses = np.zeros(R)
            for r in range(R):
                if isinstance(U[r][0][0], (list, tuple, np.ndarray)):
                    mses[r] = np.mean(orderpartitions(U[r], method=method, X=X, GDM=GDM)[1])
                else:
                    mses[r] = np.mean([mn.mseclustersfuzzy(X, U[r], donormalise=False, GDM=GDM)])
            order = np.argsort(mses)
            return order, mses[order]

    # Fix parameters
    Uloc = ds.listofarrays2arrayofarrays(U)
    R = len(Uloc)
    if GDM is None:
        GDMloc = np.ones([Uloc[0].shape[0], R], dtype=bool)
    elif GDM.shape[1] == 1:
        if R > 1:
            GDMloc = np.tile(GDM, [1, R])
        else:
            GDMloc = np.array(GDM)
    else:
        GDMloc = np.array(GDM)
    if w is None or (w is str and w in ['all', 'equal']):
        w = np.ones(R)
    elif ds.numel(w) == 1:
        w = np.array([w for i in range(R)])
    wmeans = calwmeans(w)

    # Work!
    #permR = orderpartitions(Uloc, method='rand', X=X, GDM=GDM)[0]
    if GDM is None:
        permR = orderpartitions(Uloc, method='mse', X=X, GDM=None)[0]
    else:
        permR = orderpartitions(Uloc, method='mse', X=X, GDM=GDMloc)[0]
    Uloc = Uloc[permR]
    if GDMloc.shape[1] > 1:
        GDMloc = GDMloc[:,permR]
    wmeans = wmeans[permR]

    if isinstance(Uloc[0][0][0], (list, tuple, np.ndarray)):
        Uloc[0] = generateCoPaM(Uloc[0], relabel_technique=relabel_technique, w=w[0], X=X, distCriterion=distCriterion,
                                K=K, GDM=GDMloc)
    #CoPaM = np.zeros([GDMloc.shape[0], Uloc[0].shape[1]], float)
    CoPaM = np.array(Uloc[0], dtype=float)
    K = CoPaM.shape[1]
    for r in range(1,R):
        if isinstance(Uloc[r][0][0], (list, tuple, np.ndarray)):
            Uloc[r] = generateCoPaM(Uloc[r], relabel_technique=relabel_technique, w=w[r], X=X,
                                    distCriterion=distCriterion, K=K, GDM=GDMloc)
        if Uloc[r].shape[1] != K:
            raise ValueError('Inequal numbers of clusters in the partition {}.'.format(r))

        Uloc[r] = relabelClusts(CoPaM, Uloc[r], method=relabel_technique, X=X,
                                distCriterion=distCriterion)

        dotprod = np.dot(GDMloc[:, 0:r], wmeans[0:r].transpose())  # (Mxr) * (rx1) = (Mx1)
        CoPaM[dotprod > 0] = nu.multiplyaxis(CoPaM[dotprod > 0], dotprod[dotprod > 0], axis=1)
        CoPaM[dotprod > 0] += wmeans[r] * Uloc[r][dotprod > 0]
        dotprod = np.dot(GDMloc[:, 0:(r + 1)], wmeans[0:(r + 1)].transpose())
        CoPaM[dotprod > 0] = nu.divideaxis(CoPaM[dotprod > 0], dotprod[dotprod > 0], axis=1)

    return CoPaM


def generateCoPaMfromidx(U, relabel_technique='minmin', w=None, X=None, distCriterion='direct_euc', K=0, GDM=None):
    # TODO generate CoPaM from idx
    raise NotImplementedError()


def sortclusters(CoPaM, Mc, minGenesinClust = 11):
    Mcloc = np.array(Mc)
    [Np, K] = Mcloc.shape
    largerThanMax = np.max(Mcloc) + 1
    Cf = np.zeros(K, dtype=int) - 1

    for i in range(Np-1,-1,-1):
        C = np.argsort(Mcloc[i])[::-1]
        M = Mcloc[i,C]
        Cf[np.all([M >= minGenesinClust, Cf == 0], axis=0)] = C[np.all([M >= minGenesinClust, Cf == 0], axis=0)]
        if i > 0:
            Mcloc[i-1, Cf[Cf != 0]] = largerThanMax

    Cf[Cf==-1] = np.setdiff1d(np.arange(K), Cf)

    return np.array(CoPaM)[:, Cf]


# Clustering helping function for parallel loop
def clustDataset(X, K, D, methods, GDMcolumn, Ng):
    #X, K, D, methods, GDMcolumn, Ng = args
    Uloc = [np.zeros([Ng, K], dtype=bool)] * len(methods)  # Prepare the U output
    tmpU = cl.clusterdataset(X, K, D, methods)  # Obtain U's
    for cc in range(len(methods)):  # Set U's as per the GDM values
        Uloc[cc][GDMcolumn] = tmpU[cc]
    return Uloc


def uncles(X, type='A', Ks=[n for n in range(2, 21)], params=None, methods=None, methodsDetailed=None, U=None,
           Utype='PM', relabel_technique='minmin', setsP=None, setsN=None, dofuzzystretch=False, wsets=None,
           wmethods=None, GDM=None, smallestClusterSize=11, CoPaMfinetrials=1, CoPaMfinaltrials=1,
           binarise_techniqueP='DTB', binarise_paramP=np.arange(0.0,1.1,0.1,dtype='float'), binarise_techniqueN='DTB',
           binarise_paramN=np.concatenate(([sys.float_info.epsilon], np.arange(0.1,1.1,0.1,dtype='float'))),
           Xnames=None, ncores=1):
    Xloc = ds.listofarrays2arrayofarrays(X)
    L = len(Xloc)  # Number of datasets

    # Fix parameters
    if params is None: params = {}
    if setsP is None: setsP = [x for x in range(int(math.floor(L / 2)))]
    if setsN is None: setsN = [x for x in range(int(math.floor(L / 2)), L)]
    setsPN = np.array(np.concatenate((setsP, setsN), axis=0), dtype=int)
    Xloc = Xloc[setsPN]
    L = np.shape(Xloc)[0]  # Number of datasets
    if methods is None: methods = [['k-means'], ['SOMs'], ['HC', 'linkage_method', 'ward']]
    #if methods is None: methods = [['k-means'], ['HC', 'linkage_method', 'ward']]
    #if methods is None: methods = [['HC', 'linkage_method', 'ward']]
    if methodsDetailed is None:
        methodsDetailedloc = np.array([methods for l in range(L)])
        #methodsDetailedloc = np.tile(methods, [L,1])
    else:
        methodsDetailedloc = methodsDetailed[setsPN]
    if wsets is None:
        wsets = np.array([1 for x in range(L)])
    else:
        wsets = np.array(wsets)[setsPN]
    if wmethods is None:
        wmethods = [[1 for x in m] for m in methodsDetailedloc]
    elif not isinstance(wmethods[0], (list,tuple,np.ndarray)):
        wmethods = np.tile(methods,[L,1])
    else:
        wmethods = np.array(wmethods)[setsPN]
    if GDM is None:
        Ng = np.shape(Xloc[0])[0]
        GDMloc = np.ones([Ng, L], dtype='bool')
    else:
        GDMloc = GDM[:, setsPN]
        Ng = GDMloc.shape[0]
    if Xnames is None:
        Xnames = ['X{0}'.format(l) for l in range(L)]

    setsPloc = [ii for ii in range(len(setsP))]
    if L > len(setsPloc):
        setsNloc = [ii for ii in range(len(setsPloc),L)]

    Ds = [nu.closest_to_square_factors(k) for k in Ks]  # Grid sizes for the SOMs method for each value of K
    NKs = len(Ks)  # Number of K values

    # Clustering
    if U is None:
        Utype = 'PM'
        Uloc = np.array([None] * (L * NKs)).reshape([L, NKs])
        io.resetparallelprogress(np.sum(Ks) * np.sum([len(meths) for meths in methodsDetailedloc]))

        for l in range(L):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Utmp = Parallel(n_jobs=ncores)\
                    (delayed(clustDataset)
                     (Xloc[l], Ks[ki], Ds[ki], methodsDetailedloc[l], GDMloc[:, l], Ng) for ki in range(NKs))

                Utmp = [u for u in Utmp]
                for ki in range(NKs):
                    Uloc[l, ki] = Utmp[ki]

                gc.collect()
                #io.updateparallelprogress(np.sum(Ks) * len(methodsDetailedloc))

        '''
    elif U is None:
        Utype = 'PM'
        Uloc = np.array([None] * (L * NKs)).reshape([L, NKs])
        for l in range(L):
            for ki in range(NKs):
                io.log('Cluster {0}, K = {1}'.format(Xnames[l], Ks[ki]))
                Uloc[l, ki] = [np.zeros([Ng,Ks[ki]], dtype=bool)] * len(methodsDetailedloc[l])  # Prepare the U output
                tmpU = cl.clusterdataset(Xloc[l], Ks[ki], Ds[ki], methodsDetailedloc[l])  # Obtain U's
                for cc in range(len(methodsDetailedloc[l])):  # Set U's as per the GDM values
                    Uloc[l, ki][cc][GDMloc[:, l]] = tmpU[cc]
        '''
    else:
        Uloc = ds.listofarrays2arrayofarrays(U)[setsPN]

    # Calculate a CoPaM for each dataset at each K
    CoPaMsFine = np.array([None] * (L * NKs)).reshape([L, NKs])
    for l in range(L):
        for ki in range(NKs):
            if Utype.lower() == 'pm':
                CoPaMsFineTmp = [generateCoPaM(Uloc[l,ki],relabel_technique=relabel_technique, X=[Xloc[l]],
                                               w=wmethods[l], K=Ks[ki], GDM=GDMloc[:,l].reshape([-1,1]))
                                 for i in range(CoPaMfinetrials)]
            elif Utype.lower() == 'idx':
                CoPaMsFineTmp = \
                    [generateCoPaMfromidx(Uloc[l, ki], relabel_technique=relabel_technique, X=Xloc,
                                          w=wmethods[l], K=Ks[ki])
                     for i in range(CoPaMfinetrials)]
            else:
                raise ValueError('Invalid Utype')
            CoPaMsFine[l, ki] = generateCoPaM(CoPaMsFineTmp, relabel_technique=relabel_technique, X=[Xloc[l]],
                                              GDM=GDMloc[:, l].reshape([-1, 1]))

            if dofuzzystretch:
                CoPaMsFine[l, ki] = fuzzystretch(CoPaMsFine[l, ki])

    # Calculate the final CoPaM for each K
    CoPaMs = np.array([None] * (CoPaMfinaltrials * NKs)).reshape([CoPaMfinaltrials, NKs])
    CoPaMsP = np.array([None] * (CoPaMfinaltrials * NKs)).reshape([CoPaMfinaltrials, NKs])
    CoPaMsN = np.array([None] * (CoPaMfinaltrials * NKs)).reshape([CoPaMfinaltrials, NKs])
    for t in range(CoPaMfinaltrials):
        for ki in range(NKs):
            if type == 'A':
                if Utype.lower() == 'pm':
                    CoPaMs[t, ki] = generateCoPaM(CoPaMsFine[:, ki], relabel_technique=relabel_technique, w=wsets,
                                                  X=Xloc, GDM=GDMloc)
                elif Utype.lower() == 'idx':
                    CoPaMs[t, ki] = generateCoPaMfromidx(CoPaMsFine[:, ki], relabel_technique=relabel_technique,
                                                         X=Xloc, w=wsets, GDM=GDMloc)
                else:
                    raise ValueError('Invalid Utype')
            elif type == 'B':
                if Utype.lower() == 'pm':
                    CoPaMsP[t, ki] = generateCoPaM(CoPaMsFine[setsPloc, ki], relabel_technique=relabel_technique,
                                                   X=Xloc, w=wsets[setsPloc], GDM=GDMloc[:, setsPloc])
                    CoPaMsN[t, ki] = generateCoPaM(CoPaMsFine[setsNloc, ki], relabel_technique=relabel_technique,
                                                   X=Xloc, w=wsets[setsNloc], GDM=GDMloc[:, setsNloc])
                elif Utype.lower() == 'idx':
                    CoPaMsP[t, ki] = generateCoPaMfromidx(CoPaMsFine[setsPloc, ki], relabel_technique=relabel_technique,
                                                          X=Xloc, w=wsets[setsPloc], GDM=GDMloc[:, setsPloc])
                    CoPaMsN[t, ki] = generateCoPaMfromidx(CoPaMsFine[setsNloc, ki], relabel_technique=relabel_technique,
                                                          X=Xloc, w=wsets[setsNloc], GDM=GDMloc[:, setsNloc])
                else:
                    raise ValueError('Invalid Utype')
            else:
                raise ValueError('Invalid UNCLES type. It has to be either A or B')


    # Binarise
    NPp = len(binarise_paramP) # Number of P params
    NNp = len(binarise_paramN) # Number of N params
    if type == 'A':
        B = np.zeros([CoPaMfinaltrials, NPp, 1, NKs], dtype=object)
        Mc = np.zeros([CoPaMfinaltrials, NKs], dtype=object)
    elif type == 'B':
        B = np.zeros([CoPaMfinaltrials, NPp, NNp, NKs], dtype=object)
        Mc = np.zeros([CoPaMfinaltrials, NKs], dtype=object)

    for t in range(CoPaMfinaltrials):
        for ki in range(NKs):
            if type =='A':
                # Pre-sorting binarisation
                for p in range(NPp):
                    B[t,p,0,ki] = binarise(CoPaMs[t, ki], binarise_techniqueP, binarise_paramP[p])
                Mc[t, ki] = [np.sum(Bp, axis=0) for Bp in B[t,:,0,ki]]

                # Sorting
                CoPaMs[t, ki] = sortclusters(CoPaMs[t, ki], Mc[t, ki], smallestClusterSize)

                # Post-sorting binarisation
                for p in range(NPp):
                    B[t,p,0,ki] = binarise(CoPaMs[t, ki], binarise_techniqueP, binarise_paramP[p])
                Mc[t, ki] = [np.sum(Bp, axis=0) for Bp in B[t,:,0,ki]]
            elif type == 'B':
                # Pre-sorting binarisation
                BP = [binarise(CoPaMsP[t, ki], binarise_techniqueP, binarise_paramP[p]) for p in range(NPp)]
                McP = [np.sum(BPp, axis=0) for BPp in BP]

                BN = [binarise(CoPaMsN[t, ki], binarise_techniqueN, binarise_paramN[p]) for p in range(NNp)]
                McN = [np.sum(BNp, axis=0) for BNp in BN]

                # Sorting
                CoPaMsP[t, ki] = sortclusters(CoPaMsP[t, ki], McP, smallestClusterSize)
                CoPaMsN[t, ki] = sortclusters(CoPaMsN[t, ki], McN, smallestClusterSize)

                # Post-sorting binarisation
                BP = [binarise(CoPaMsP[t, ki], binarise_techniqueP, binarise_paramP[p]) for p in range(NPp)]
                McP = [np.sum(BPp, axis=0) for BPp in BP]

                BN = [binarise(CoPaMsN[t, ki], binarise_techniqueN, binarise_paramN[p]) for p in range(NNp)]
                McN = [np.sum(BNp, axis=0) for BNp in BN]

                # UNCLES B logic
                for pp in range(NPp):
                    for pn in range(NNp):
                        B[t,pp,pn,ki] = BP[pp]
                        B[t,pp,pn,ki][np.any(BN[pn], axis=1)] = False

                # Fill Mc
                Mc[t, ki] = [None] * Ks[ki]
                for k in range(Ks[ki]):
                    Mc[t, ki][k] = np.zeros([NPp, NNp])
                    for pp in range(NPp):
                        for pn in range(NNp):
                            Mc[t, ki][k][pp, pn] = np.sum(B[t,pp,pn,ki][:,k])

    # Prepare and return the results:
    params = dict(params, **{
        'methods': methods,
        'setsP': setsPloc,
        'setsN': setsNloc,
        'dofuzzystretch': dofuzzystretch,
        'type': type,
        'Ks': Ks,
        'NKs': NKs,
        'wsets': wsets,
        'wmethods': wmethods,
        'Ds': Ds,
        'L': L,
        'CoPaMs': CoPaMs,
        'smallestclustersize': smallestClusterSize,
        'GDM': GDMloc
    })

    UnclesRes = collections.namedtuple('UnclesRes', ['B', 'Mc', 'params', 'X', 'U'])
    return UnclesRes(B, Mc, params, Xloc, Uloc)









