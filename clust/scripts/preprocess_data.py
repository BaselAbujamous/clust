import numpy as np
import scipy.interpolate as spinter
import scipy.stats.mstats as spmstats
import math
import datastructures as ds
import numeric as nu
import re
import io
import warnings
from copy import deepcopy


def isnan(X):
    if ds.numel(X) == 1:
        return math.isnan(X)
    elif len(np.shape(X)) == 1:
        res = np.zeros(np.shape(X), dtype=bool)
        for i in range(len(X)):
                res[i] = math.isnan(X[i])
        return res
    elif len(np.shape(X)) == 2:
        res = np.zeros(np.shape(X), dtype=bool)
        for i in range(np.size(X, 0)):
            for j in range(np.size(X, 1)):
                res[i, j] = math.isnan(X[i, j])
        return res


def fixnans(Xin, method='spline'):
    def fixrow(rowin, method='spline'):
        rowout = np.array(rowin)

        unknown = isnan(rowin)
        known = [not x for x in unknown]

        tknown = np.nonzero(known)[0]
        tunknown = np.nonzero((unknown))[0]

        xknown = np.take(rowin, tknown)

        if method == 'spline':
            if len(xknown) > 3:
                sf = spinter.UnivariateSpline(tknown, xknown)
            else:
                sf = spinter.UnivariateSpline(tknown, xknown, k=len(xknown)-1)
            rowout[tunknown] = sf(tunknown)
        else:
            raise ValueError('Provided interpolation method is not supported')

        return rowout

    Xinloc = deepcopy(Xin)
    N = np.size(Xinloc, 0)
    M = np.size(Xinloc, 1)
    Xout = np.zeros([N, M])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i in range(N):
            sumnans = sum(isnan(Xinloc[i]))
            notnans = [x for x in Xinloc[i] if not isnan(x)]
            if sumnans < M - 1:
                if math.isnan(Xinloc[i, 0]):
                    Xinloc[i, 0] = notnans[0]
                if math.isnan(Xinloc[i, -1]):
                    Xinloc[i, -1] = notnans[-1]
                Xout[i] = fixrow(Xinloc[i], method)
            elif sumnans == M - 1:
                Xout[i] = [notnans[0] for x in range(M)]
    return Xout


def normaliseSampleFeatureMat(X, type):
    """
    X = normalizeSampleFeatureMat(X, type)

    type: 0 (none), 1 (divide by mean), 2 (divide by the first),
        3 (take log2), 4 (subtract the mean and divide by the std),
        5 (divide by the sum), 6 (subtract the mean),
        7 (divide by the max), 8 (2 to the power X), 9 (subtract the min),
        10 (rank: 1 for lowest, then 2, 3, ...; average on ties),
        11 (rank, like 10 but order arbitrarly on ties),
        12 (normalise to the [0 1] range)

        101 (quantile), 102 (subtract columns (samples) means),
        103 (subtract global mean)

    If (type) was a vector like [3 1], this means to apply normalisation
    type (3) over (X) then to apply type (1) over the result. And so on.

    :param X:
    :param type:
    :return:
    """
    Xout = np.array(X)
    if isinstance(type, (list, tuple, np.ndarray)):
        for i in type:
            Xout = normaliseSampleFeatureMat(Xout, i)
        return Xout

    if type == 1:
        #1: Divide by the mean
        Xout = nu.divideaxis(Xout, np.mean(Xout, axis=1), 1)

    if type == 2:
        #2: Divide by the first value
        Xout = nu.divideaxis(Xout, Xout[:,1], 1)

    if type == 3:
        #3: Take log2
        Xout[Xout<=0] = float('nan')
        Xout = np.log2(Xout)
        ind1 = np.any(isnan(Xout), axis=1)
        Xout[ind1] = fixnans(Xout[ind1])

    if type == 4:
        #4: Subtract the mean and divide by the std
        Xout = nu.subtractaxis(Xout, np.mean(Xout, axis=1), axis=1)
        ConstGenesIndices = np.std(Xout, axis=1) == 0
        Xout = nu.divideaxis(Xout, np.std(Xout, axis=1), axis=1)
        Xout[ConstGenesIndices] = 0

    if type == 5:
        #5: Divide by the sum
        Xout = nu.divideaxis(Xout, np.sum(Xout, axis=1), axis=1)

    if type == 6:
        #6: Subtract the mean
        Xout = nu.subtractaxis(Xout, np.mean(Xout, axis=1), axis=1)

    if type == 7:
        #7: Divide by the maximum
        Xout = nu.divideaxis(Xout, np.max(Xout, axis=1), axis=1)

    if type == 8:
        #8: (2 to the power X)
        Xout = np.power(2, Xout)

    if type == 9:
        #9: Subtract the min
        Xout = nu.subtractaxis(Xout, np.min(Xout, axis=1), axis=1)

    if type == 10:
        #10: Rank: 0 for lowest, then 1, 2, ...; average on ties
        Xout = spmstats.rankdata(Xout, axis=0) - 1

    if type == 11:
        #11: Rank: 0 for lowest, then 1, 2, ...; arbitrary order on ties
        Xout = np.argsort(np.argsort(Xout,axis=0), axis=0)

    if type == 12:
        #12: Normalise to the [0 1] range
        Xout = nu.subtractaxis(Xout, np.min(Xout, axis=1), axis=1)
        Xout = nu.divideaxis(Xout, np.max(Xout, axis=1), axis=1)

    # 100s
    if type == 101:
        #101: quantile
        av = np.mean(np.sort(Xout, axis=0), axis=1)
        II = np.argsort(np.argsort(Xout, axis=0), axis=0)
        Xout = av[II]

    if type == 102:
        #102: subtract the mean of each sample (column) from it
        Xout = nu.subtractaxis(Xout, np.mean(Xout,axis=0), axis=0)

    if type == 103:
        #103: subtract the global mean of the data
        Xout -= np.mean(Xout)

    return Xout


def mapGenesToCommonIDs(Genes, Map, mapheader=True, OGsFirstColMap=True, delimGenesInMap='W+'):
    L = len(Genes)  # Number of datasets (i.e. lists of gene names)
    Maploc = np.array(Map, dtype=object)
    if mapheader:
        MapSpecies = Maploc[0]
        Maploc = Maploc[1:]
    else:
        MapSpecies = None

    # If the OG IDs are given in the Map, use them; otherwise generate them as OG0000000 to OGxxxxxxx
    if OGsFirstColMap:
        OGs = Maploc[:, 0].flatten()
        Maploc = Maploc[:, 1:]
        if MapSpecies is None:
            MapSpecies = np.array(['Species{}'.format(i) for i in range(Maploc.shape[1])])
        else:
            MapSpecies = MapSpecies[1:]
    else:
        OGs = np.array(['OG%07d' % i for i in range(Maploc.shape[0])])

    # !!!!!!!!TRANSPOSE MAP!!!!!!!!
    Maploc = Maploc.transpose()  # Now this is: Maploc[species][gene]

    # Split Map entries by the delim
    for i in range(Maploc.shape[0]):
        for j in range(Maploc.shape[1]):
            Maploc[i, j] = re.split(delimGenesInMap, Maploc[i, j])

    # Generate a flattened version of the Map: FlattenedMap[s] is a 1d list of all genes in the (s)th Map row, i.e.
    # in the (s)th species; this will make FlattenedMap[s1][n] not necessarily corresponding to FlattenedMap[s2][n])
    S = Maploc.shape[0]  # Number of species
    FlattenedMap = [np.array(ds.flattenAList(ms.tolist())) for ms in Maploc]

    OGsDatasets = np.array([None] * L, dtype=object)
    for l in range(L):
        Ng = len(Genes[l])  # Number of genes in this dataset
        s = np.argmax([len(np.intersect1d(Genes[l], speciesgenes))
                       for speciesgenes in FlattenedMap])  # The most matching species

        OGsDatasets[l] = np.array(['' for i in range(Ng)], dtype=object)  # Default gene name for unmapped genes is ''
        findGenesInMap = ds.findArrayInSubArraysOfAnotherArray1D(Genes[l], Maploc[s])  # Indices of Genes in the Map (Ngx1 indices)
        OGsDatasets[l][findGenesInMap > -1] = OGs[findGenesInMap[findGenesInMap > -1]]


    OGsFiltered = np.unique(ds.flattenAList(OGsDatasets.flatten().tolist()))  # Get sorted unique and *USED* OGs
    OGsFiltered = OGsFiltered[OGsFiltered != '']
    I = ds.findArrayInAnotherArray1D(OGsFiltered, OGs)
    Maploc = Maploc.transpose()[I]

    # Return
    return (OGsFiltered, OGsDatasets, Maploc, MapSpecies)


def calculateGDMandUpdateDatasets(X, Genes, Map=None, mapheader=True, OGsFirstColMap=True, delimGenesInMap='W+', OGsIncludedIfAtLeastInDatasets=1):
    Xloc = ds.listofarrays2arrayofarrays(X)
    Genesloc = deepcopy(Genes)
    if Map is None:
        OGsDatasets = deepcopy(Genes)
        OGs = np.unique(ds.flattenAList(OGsDatasets))  # Unique list of genes (or mapped genes)
        MapNew = None
        MapSpecies = None
    else:
        (OGs, OGsDatasets, MapNew, MapSpecies) = mapGenesToCommonIDs(Genes, Map, mapheader, OGsFirstColMap, delimGenesInMap)

    L = len(Genesloc)  # Number of datasets
    Ng = len(OGs)  # Number of unique genes

    GDMall = np.transpose([np.in1d(OGs, gs) for gs in OGsDatasets])  # GDM: (Ng)x(L) boolean

    # Exclude OGs that do not exist in at least (OGsIncludedIfAtLeastInDatasets) datasets
    IncludedOGs = np.sum(GDMall, axis=1) >= OGsIncludedIfAtLeastInDatasets
    GDM = GDMall[IncludedOGs]
    OGs = OGs[IncludedOGs]

    Ngs = np.sum(GDM, axis=0)  # Numbers of unique mapped genes in each dataset

    Xnew = np.array([None] * L, dtype=object)
    GenesDatasets = np.array([None] * L, dtype=object)
    for l in range(L):
        d = Xloc[l].shape[1]  # Number of dimensions (samples) in this dataset
        Xnew[l] = np.zeros([Ngs[l], d], dtype=float)
        GenesDatasets[l] = np.empty(Ngs[l], dtype=object)
        OGsInThisDS = OGs[GDM[:, l]]  # Unique OGs in this dataset
        # TODO: Optimise the code below by exploiting ds.findArrayInSubArraysOfAnotherArray1D (like in line 203 above)
        for ogi in range(len(OGsInThisDS)):
            og = OGsInThisDS[ogi]
            Xnew[l][ogi] = np.sum(Xloc[l][np.in1d(OGsDatasets[l], og)], axis=0)
            GenesDatasets[l][ogi] = ds.concatenateStrings(Genesloc[l][np.in1d(OGsDatasets[l], og)])

    return (Xnew, GDM, GDMall, OGs, MapNew, MapSpecies)


def filterlowgenes(X, GDM, threshold=10.0, replacementVal=0.0, atleastinconditions=1, atleastindatasets=1):
    Xloc = np.array(X)
    GDMloc = np.array(GDM)
    L = len(Xloc)  # Number of the datasets
    Ng = GDMloc.shape[0]  # Number of genes

    # Set values less than the threshold to zero, then
    # find genes which do not pass this threshold, i.e. have been set to zero, at all:
    Iincluded = np.zeros([Ng, L], dtype=bool)  # The genes which pass the threshold in at least atleastinconditions
    for l in range(L):
        Xloc[l][Xloc[l] < threshold] = replacementVal
        Iincluded[GDMloc[:, l], l] = np.sum(Xloc[l] >= threshold, axis=1) >= atleastinconditions
    Iincluded = np.sum(Iincluded, axis=1) >= atleastindatasets

    # Update Xloc, Genesloc, and finally GDM loc
    for l in range(L):
        Xloc[l] = Xloc[l][Iincluded[GDMloc[:, l]]]
    GDMloc = GDMloc[Iincluded]

    # Return results:
    return (Xloc, GDMloc, Iincluded)


def combineReplicates(X, replicatesIDs, flipSamples):
    Xloc = np.array(X)
    L = len(Xloc)

    for l in range(L):
        Xtmp = Xloc[l]
        if flipSamples is not None and flipSamples[l] is not None and len(flipSamples[l]) == Xtmp.shape[1]:
            Xtmp[:, flipSamples[l] == 1] = np.divide(1.0, Xtmp[:, flipSamples[l] == 1])
            Xtmp[:, flipSamples[l] == 2] = -Xtmp[:, flipSamples[l] == 2]
        uniqueSamples = np.unique(replicatesIDs[l])
        uniqueSamples = uniqueSamples[uniqueSamples != -1]
        Xloc[l] = np.zeros([Xtmp.shape[0], len(uniqueSamples)])
        ss = 0
        for s in range(len(uniqueSamples)):
            if uniqueSamples[s] > -1:
                Xloc[l][:, ss] = np.median(Xtmp[:, replicatesIDs[l] == uniqueSamples[s]], axis=1)
                ss += 1

    return Xloc


def preprocess(X, GDM, normalise=0, replicatesIDs=None, flipSamples=None, expressionValueThreshold=10.0,
               replacementVal=0.0, atleastinconditions=1, atleastindatasets=1, params=None):
    # Fixing parameters
    Xloc = ds.listofarrays2arrayofarrays(X)
    L = len(Xloc)
    if params is None: params = {}
    if replicatesIDs is None:
        replicatesIDsloc = [np.array([ii for ii in range(x.shape[1])]) for x in Xloc]
    else:
        replicatesIDsloc = ds.listofarrays2arrayofarrays(replicatesIDs)
        replicatesIDsloc = [np.array(x) for x in replicatesIDsloc]
    if flipSamples is None:
        flipSamplesloc = None
    else:
        flipSamplesloc = ds.listofarrays2arrayofarrays(flipSamples)
        flipSamplesloc = [np.array(x) for x in flipSamplesloc]
    if not isinstance(normalise, (list, tuple, np.ndarray)):
        normaliseloc = [normalise if isinstance(normalise, (list, tuple, np.ndarray)) else [normalise] for i in range(L)]
        normaliseloc = ds.listofarrays2arrayofarrays(normaliseloc)
    else:
        normaliseloc = [nor if isinstance(nor, (list, tuple, np.ndarray)) else [nor] for nor in normalise]
        normaliseloc = ds.listofarrays2arrayofarrays(normaliseloc)

    # Get rid of nans by fixing
    Xproc = Xloc
    for l in range(L):
        Xproc[l] = fixnans(Xproc[l])

    # Combine replicates and sort out flipped samples
    Xproc = combineReplicates(Xproc, replicatesIDsloc, flipSamplesloc)

    # Filter genes not exceeding the threshold
    (Xproc, GDMnew, Iincluded) = filterlowgenes(Xproc, GDM, expressionValueThreshold, replacementVal,
                                                atleastinconditions, atleastindatasets)

    # Normalise
    for l in range(L):
        Xproc[l] = normaliseSampleFeatureMat(Xproc[l], normaliseloc[l])

    # Prepare params for the output
    params = dict(params, **{
        'normalise': normaliseloc,
        'replicatesIDs': replicatesIDs,
        'flipSamples': flipSamplesloc,
        'L': L
    })

    return (Xproc, GDMnew, Iincluded, params)


