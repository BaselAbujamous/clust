import numpy as np
import scipy.interpolate as spinter
import scipy.stats.mstats as spmstats
import scipy.stats as spst
import sklearn.mixture as skmix
import math
import datastructures as ds
import numeric as nu
import re
import warnings
from copy import deepcopy
import glob
import output as op
import io
import collections as collec


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
    def fixrow(rowin, methodloc='spline'):
        rowout = np.array(rowin)

        unknown = isnan(rowin)
        known = [not xx for xx in unknown]

        tknown = np.nonzero(known)[0]
        tunknown = np.nonzero(unknown)[0]

        xknown = np.take(rowin, tknown)

        if methodloc == 'spline':
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


def percentage_less_than(X, v):
    """
    Percentage of elements in matrix X that are less than the value v
    :param X: Matrix of numbers (numpy array)
    :param v: A value to be compared with
    :return: A percentage in the range [0.0, 1.0]
    """
    return np.sum(X < v) * 1.0 / ds.numel(X)


def isnormal_68_95_99p7_rule(X):
    """
    Test if data is normally distributed by checking the percentages of values below different stds away from the mean
    This is not fully implemented and is not used in the current version of the method
    :param X: Dataset matrix (numpy array)
    :return:
    """
    n = ds.numel(X)
    m = np.mean(X)
    s = np.std(X)

    bins = np.linspace(np.min(X), np.max(X), 100)
    d = np.digitize(np.concatenate(X), bins)
    xd = bins[d-1]
    mode = spst.mode(xd)[0]

    # Find the percentage of elements less than these seven values
    m3s = percentage_less_than(X, m - 3 * s)  # mean minus 3s (theory ~= N(0.0013, s=0.0315/sqrt(n)))
    m2s = percentage_less_than(X, m - 2 * s)  # mean minus 2s (theory ~= N(0.0228, s=0.1153/sqrt(n)))
    m1s = percentage_less_than(X, m - 1 * s)  # mean minus 1s (theory ~= N(0.1587, s=0.2116/sqrt(n)))
    p0s = percentage_less_than(X, m)  # mean (theory ~= N(0.5000, s=0.3013/sqrt(n)))
    p1s = percentage_less_than(X, m + 1 * s)  # mean plus 1s (theory ~= N(0.8413, s=0.2116/sqrt(n)))
    p2s = percentage_less_than(X, m + 2 * s)  # mean plus 2s (theory ~= N(0.9772, s=0.1153/sqrt(n)))
    p3s = percentage_less_than(X, m + 3 * s)  # mean plus 3s (theory ~= N(0.9987, s=0.0315/sqrt(n)))
    md = percentage_less_than(X, mode)  # mode (theory ~= N(0.9987, s=0.0315/sqrt(n)))

    # How were these theoretical distributions calculated??
    # The distributions of these stds were found empirically by calculating them from 1000x26 randomly generated
    # normally distributed numbers ~N(0.0, 1.0). 26 different population sizes were considered "round(10.^(1:0.2:6))",
    # at each population size, 1000 random populations were generated. It was observed that at a fixed population size,
    # the percentages of elements less than (m-3*s) or (m-2*s) ... (etc.) were normally distributed with an average
    # equal to the expected CDF at (m-3*s) or (m-2*s) ... (etc.) and with a standard deviation that is inversely
    # linearly proportional to the square root of the size of the population. The empirical values were calculated from
    # this experiment and are included above. For example: the percentage of elements that are less than (m-2*s) in a
    # population of n elements is expected to be 0.0228 (2.28%) with a standard deviation of 0.1587/sqrt(n).
    # This empirical test was run on MATLAB

    # Calculate one-tailed p-values for the seven values above based on normal distributions
    pv = np.array([i*1.0 for i in range(8)])
    diff = np.array([i*1.0 for i in range(8)])

    pv[0] = 1-2*spst.norm.cdf(-abs(m3s-0.0013), loc=0, scale=0.0315/math.sqrt(n))
    diff[0] = abs(m3s-0.0013)

    pv[1] = 1-2*spst.norm.cdf(-abs(m2s-0.0228), loc=0, scale=0.1153/math.sqrt(n))
    diff[1] = abs(m2s-0.0228)

    pv[2] = 1-2*spst.norm.cdf(-abs(m1s-0.1587), loc=0, scale=0.2116/math.sqrt(n))
    diff[2] = abs(m1s-0.1587)

    pv[3] = 1-2*spst.norm.cdf(-abs(p0s-0.5000), loc=0, scale=0.3013/math.sqrt(n))
    diff[3] = abs(p0s-0.5000)

    pv[4] = 1-2*spst.norm.cdf(-abs(p1s-0.8413), loc=0, scale=0.2116/math.sqrt(n))
    diff[4] = abs(p1s-0.8413)

    pv[5] = 1-2*spst.norm.cdf(-abs(p2s-0.9772), loc=0, scale=0.1153/math.sqrt(n))
    diff[5] = abs(p2s-0.9772)

    pv[6] = 1-2*spst.norm.cdf(-abs(p3s-0.9987), loc=0, scale=0.0315/math.sqrt(n))
    diff[6] = abs(p3s-0.9987)

    pv[7] = 1 - 2 * spst.norm.cdf(-abs(md - 0.5000), loc=0, scale=0.3013 / math.sqrt(n))
    diff[7] = abs(md - 0.5000)

    return np.mean(np.log10(pv)), np.mean(diff)


def filterBimodal(X):
    """
    Filter out genes that always have low expression values
    :param X: Dataset matrix (numpy array)
    :return: Tuple: (Filtered X, Iincluded)
    """
    xmax = np.max(X, axis=1)
    xmaxsort = xmax[xmax > 0]  # take out zero values
    xmaxsort = xmaxsort[np.all([xmaxsort > np.percentile(xmaxsort, q=1), xmaxsort < np.percentile(xmaxsort, q=99)],
                               axis=0)]  # take out 1% big & small outliers
    xmaxsort = np.sort(xmaxsort).reshape(-1, 1)
    GM = skmix.GaussianMixture(n_components=2)
    GM.fit(xmaxsort)
    labels = GM.predict(xmaxsort)
    th1 = np.max(xmaxsort[labels == labels[0]])

    Iexcluded = xmax < th1
    Xf = np.array(X)
    Xf[Iexcluded] = 0.0

    return Xf


def autoNormalise(X):
    """
    Automatically normalise dataset X and filter it if needed

    :param X: Dataset matrix (numpy array)
    :return: array of normalisation codes
    """
    Xloc = np.array(X)

    twosided = np.sum(Xloc < 0) > 0.2 * np.sum(Xloc > 0)  # negative values are at least 20% of positive values
    alreadylogs = np.sum(abs(Xloc) < 30) > 0.98 * ds.numel(Xloc)  # More than 98% of values are below 30.0

    if twosided:
        return np.array([6])
        #return np.array([101, 4])
    else:
        Xloc[isnan(Xloc)] = 0.0
        Xloc[Xloc < 0] = 0.0
        if alreadylogs:
            Xf = normaliseSampleFeatureMat(Xloc, [13])[0]
            if isnormal_68_95_99p7_rule(Xf)[1] < isnormal_68_95_99p7_rule(Xloc)[1]:
                return np.array([13, 4])
            else:
                return np.array([4])
        else:
            Xl = normaliseSampleFeatureMat(Xloc, [3])[0]  # index 1  (Xloc, i.e. original X is index 0)
            Xlp = normaliseSampleFeatureMat(Xloc, [31])[0]  # index 2
            Xf = normaliseSampleFeatureMat(Xloc, [13])[0]  # index 3
            Xlf = normaliseSampleFeatureMat(Xl, [13])[0]  # index 4
            Xlpf = normaliseSampleFeatureMat(Xlp, [13])[0]  # index 5
            isnormal_stats = [isnormal_68_95_99p7_rule(Xloc)[1], isnormal_68_95_99p7_rule(Xl)[1],
                              isnormal_68_95_99p7_rule(Xlp)[1], isnormal_68_95_99p7_rule(Xf)[1],
                              isnormal_68_95_99p7_rule(Xlf)[1], isnormal_68_95_99p7_rule(Xlpf)[1]]
            most_normal_index = np.argmin(isnormal_stats)
            if most_normal_index == 0:
                return np.array([4])
            elif most_normal_index == 1:
                return np.array([3, 4])
            elif most_normal_index == 2:
                return np.array([31, 4])
            elif most_normal_index == 3:
                return np.array([13, 4])
            elif most_normal_index == 4:
                return np.array([3, 13, 4])
            elif most_normal_index == 5:
                return np.array([31, 13, 4])
            else:
                raise ValueError('You should never reach this error. Please contact {0}'.format(glob.email))


def normaliseSampleFeatureMat(X, type):
    """
    X = normalizeSampleFeatureMat(X, type)

    type: 0 (none), 1 (divide by mean), 2 (divide by the first),
        3 (take log2), 31 (take log2 after setting all values < 1.0 to 1.0, i.e. guarantee positive log),
        4 (subtract the mean and divide by the std),
        5 (divide by the sum), 6 (subtract the mean),
        7 (divide by the max), 8 (2 to the power X), 9 (subtract the min),
        10 (rank: 1 for lowest, then 2, 3, ...; average on ties),
        11 (rank, like 10 but order arbitrarly on ties),
        12 (normalise to the [0 1] range),
        13 (Genes with low values everywhere are set to zeros; bimodel distribution is fit to maxima of rows)

        101 (quantile), 102 (subtract columns (samples) means),
        103 (subtract global mean)

        1000 (Automatically detect normalisation)

    If (type) was a vector like [3 1], this means to apply normalisation
    type (3) over (X) then to apply type (1) over the result. And so on.

    :param X:
    :param type:
    :return:
    """
    Xout = np.array(X)
    codes = np.array(type)  # stays as input types unless auto-normalisation (type 1000) changes it

    if isinstance(type, (list, tuple, np.ndarray)):
        # This has a reason, which is if there is a single type (1000), it will replace it with the actual codes
        j = 0
        for i in range(len(type)):
            Xout, codesi = normaliseSampleFeatureMat(Xout, type[i])
            if isinstance(codesi, (list, tuple, np.ndarray)) & codesi.ndim > 0:
                codes[j] = codesi[0]
                codes = np.insert(codes, j+1, codesi[1:])
                j = j + len(codesi)
            else:
                j = j + 1
        return Xout, codes

    if type == 1:
        # 1: Divide by the mean
        Xout = nu.divideaxis(Xout, np.mean(Xout, axis=1), 1)

    if type == 2:
        # 2: Divide by the first value
        Xout = nu.divideaxis(Xout, Xout[:, 1], 1)

    if type == 3:
        # 3: Take log2
        Xout[Xout <= 0] = float('nan')
        Xout = np.log2(Xout)
        ind1 = np.any(isnan(Xout), axis=1)
        Xout[ind1] = fixnans(Xout[ind1])

    if type == 31:
        # 31: Set all values < 1 to 1 then take log (guarantee a positive log)
        Xout[Xout <= 1] = 1
        Xout = np.log2(Xout)

    if type == 4:
        # 4: Subtract the mean and divide by the std
        Xout = nu.subtractaxis(Xout, np.mean(Xout, axis=1), axis=1)
        ConstGenesIndices = np.std(Xout, axis=1) == 0
        Xout = nu.divideaxis(Xout, np.std(Xout, axis=1), axis=1)
        Xout[ConstGenesIndices] = 0

    if type == 5:
        # 5: Divide by the sum
        Xout = nu.divideaxis(Xout, np.sum(Xout, axis=1), axis=1)

    if type == 6:
        # 6: Subtract the mean
        Xout = nu.subtractaxis(Xout, np.mean(Xout, axis=1), axis=1)

    if type == 7:
        # 7: Divide by the maximum
        Xout = nu.divideaxis(Xout, np.max(Xout, axis=1), axis=1)

    if type == 8:
        # 8: (2 to the power X)
        Xout = np.power(2, Xout)

    if type == 9:
        # 9: Subtract the min
        Xout = nu.subtractaxis(Xout, np.min(Xout, axis=1), axis=1)

    if type == 10:
        # 10: Rank: 0 for lowest, then 1, 2, ...; average on ties
        Xout = spmstats.rankdata(Xout, axis=0) - 1

    if type == 11:
        # 11: Rank: 0 for lowest, then 1, 2, ...; arbitrary order on ties
        Xout = np.argsort(np.argsort(Xout, axis=0), axis=0)

    if type == 12:
        # 12: Normalise to the [0 1] range
        Xout = nu.subtractaxis(Xout, np.min(Xout, axis=1), axis=1)
        Xout = nu.divideaxis(Xout, np.max(Xout, axis=1), axis=1)

    if type == 13:
        # 13: Genes with low values everywhere are set to zeros; bimodel distribution is fit to maxima of rows
        Xout = filterBimodal(X)

    # 100s
    if type == 101:
        # 101: quantile
        av = np.mean(np.sort(Xout, axis=0), axis=1)
        II = np.argsort(np.argsort(Xout, axis=0), axis=0)
        Xout = av[II]

    if type == 102:
        # 102: subtract the mean of each sample (column) from it
        Xout = nu.subtractaxis(Xout, np.mean(Xout, axis=0), axis=0)

    if type == 103:
        # 103: subtract the global mean of the data
        Xout -= np.mean(Xout)

    if type == 1000:
        # 1000: automatically detect normalisation
        codes = autoNormalise(Xout)
        Xout = normaliseSampleFeatureMat(Xout, codes)[0]
        codes = np.append([101], codes)


    return Xout, codes


def mapGenesToCommonIDs(Genes, Map, mapheader=True, OGsFirstColMap=True, delimGenesInMap='\\W+'):
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
            Maploc[i, j] = re.split(delimGenesInMap, Maploc[i, j].replace('.', 'thisisadot').replace('-', 'thisisadash').replace('/', 'thisisaslash'))
            Maploc[i, j] = [gg.replace('thisisadot', '.').replace('thisisadash', '-').replace('thisisaslash', '/') for gg in Maploc[i, j]]

    # Generate a flattened version of the Map: FlattenedMap[s] is a 1d list of all genes in the (s)th Map row, i.e.
    # in the (s)th species; this will make FlattenedMap[s1][n] not necessarily corresponding to FlattenedMap[s2][n])
    # S = Maploc.shape[0]  # Number of species
    FlattenedMap = [np.array(ds.flattenAList(ms.tolist())) for ms in Maploc]

    OGsDatasets = np.array([None] * L, dtype=object)
    for l in range(L):
        Ng = len(Genes[l])  # Number of genes in this dataset
        s = np.argmax([len(np.intersect1d(Genes[l], speciesgenes))
                       for speciesgenes in FlattenedMap])  # The most matching species

        OGsDatasets[l] = np.array(['' for i in range(Ng)], dtype=object)  # Default gene name for unmapped genes is ''
        findGenesInMap = ds.findArrayInSubArraysOfAnotherArray1D(Genes[l], Maploc[s])  # Indices of Genes in Map (Ngx1)
        findGenesInMap = findGenesInMap[:,0]  # Make it flat (length of Ng instead of array of Ngx1)
        OGsDatasets[l][findGenesInMap > -1] = OGs[findGenesInMap[findGenesInMap > -1]]

    OGsFiltered = np.unique(ds.flattenAList(OGsDatasets.flatten().tolist()))  # Get sorted unique and *USED* OGs
    OGsFiltered = OGsFiltered[OGsFiltered != '']
    I = ds.findArrayInAnotherArray1D(OGsFiltered, OGs)
    Maploc = Maploc.transpose()[I]

    # Return
    return OGsFiltered, OGsDatasets, Maploc, MapSpecies


def calculateGDMandUpdateDatasets(X, Genes, Map=None, mapheader=True, OGsFirstColMap=True, delimGenesInMap='\\W+',
                                  OGsIncludedIfAtLeastInDatasets=1):
    Xloc = ds.listofarrays2arrayofarrays(X)
    Genesloc = deepcopy(Genes)
    if Map is None:
        OGsDatasets = deepcopy(Genes)
        OGs = np.unique(ds.flattenAList(OGsDatasets))  # Unique list of genes (or mapped genes)
        MapNew = None
        MapSpecies = None
    else:
        (OGs, OGsDatasets, MapNew, MapSpecies) = mapGenesToCommonIDs(Genes, Map, mapheader,
                                                                     OGsFirstColMap, delimGenesInMap)

    L = len(Genesloc)  # Number of datasets
    # Ng = len(OGs)  # Number of unique genes

    GDMall = np.transpose([np.in1d(OGs, gs) for gs in OGsDatasets])  # GDM: (Ng)x(L) boolean

    # Exclude OGs that do not exist in at least (OGsIncludedIfAtLeastInDatasets) datasets
    IncludedOGs = np.sum(GDMall, axis=1) >= OGsIncludedIfAtLeastInDatasets
    GDM = GDMall[IncludedOGs]
    OGs = OGs[IncludedOGs]
    if MapNew is not None:
        MapNew = MapNew[IncludedOGs]

    Ngs = np.sum(GDM, axis=0)  # Numbers of unique mapped genes in each dataset

    Xnew = np.array([None] * L, dtype=object)
    GenesDatasets = np.array([None] * L, dtype=object)
    for l in range(L):
        arelogs = np.nansum(abs(Xloc[l][~isnan(Xloc[l])]) < 30) > 0.98 * ds.numel(Xloc[l][~isnan(Xloc[l])])  # More than 98% of values are below 30.0
        d = Xloc[l].shape[1]  # Number of dimensions (samples) in this dataset
        Xnew[l] = np.zeros([Ngs[l], d], dtype=float)
        GenesDatasets[l] = np.empty(Ngs[l], dtype=object)
        OGsInThisDS = OGs[GDM[:, l]]  # Unique OGs in this dataset
        # TODO: Optimise the code below by exploiting ds.findArrayInSubArraysOfAnotherArray1D (like in line 203 above)
        for ogi in range(len(OGsInThisDS)):
            og = OGsInThisDS[ogi]
            if arelogs:
                Xnew[l][ogi] = np.log2(np.sum(np.power(2.0, Xloc[l][np.in1d(OGsDatasets[l], og)]), axis=0))
            else:
                Xnew[l][ogi] = np.sum(Xloc[l][np.in1d(OGsDatasets[l], og)], axis=0)
            GenesDatasets[l][ogi] = ds.concatenateStrings(Genesloc[l][np.in1d(OGsDatasets[l], og)])

    return Xnew, GDM, GDMall, OGs, MapNew, MapSpecies


def filterlowgenes_raw(X, GDM, threshold=10.0, replacementVal=0.0, atleastinconditions=1, atleastindatasets=1,
                       absvalue=False, usereplacementval=False):
    Xloc = np.array(X)
    GDMloc = np.array(GDM)
    L = len(Xloc)  # Number of the datasets
    Ng = GDMloc.shape[0]  # Number of genes

    # Set values less than the threshold to zero, then
    # find genes which do not pass this threshold, i.e. have been set to zero, at all:
    Iincluded = np.zeros([Ng, L], dtype=bool)  # The genes which pass the threshold in at least atleastinconditions
    for l in range(L):
        if absvalue:
            if usereplacementval:
                Xloc[l][np.abs(Xloc[l]) < threshold] = replacementVal
            Iincluded[GDMloc[:, l], l] = np.sum(np.abs(Xloc[l]) >= threshold, axis=1) >= atleastinconditions
        else:
            if usereplacementval:
                Xloc[l][Xloc[l] < threshold] = replacementVal
            Iincluded[GDMloc[:, l], l] = np.sum(Xloc[l] >= threshold, axis=1) >= atleastinconditions
    Iincluded = np.sum(Iincluded, axis=1) >= atleastindatasets

    # Update Xloc, Genesloc, and finally GDM loc
    for l in range(L):
        Xloc[l] = Xloc[l][Iincluded[GDMloc[:, l]]]
    GDMloc = GDMloc[Iincluded]

    # Return results:
    return Xloc, GDMloc, Iincluded


def filterlowgenes_perc(X, GDM, threshold=10.0, replacementVal=0.0, atleastinconditions=1, atleastindatasets=1,
                        absvalue=False, usereplacementval=False):
    Xloc = np.array(X)
    GDMloc = np.array(GDM)
    L = len(Xloc)  # Number of the datasets
    Ng = GDMloc.shape[0]  # Number of genes
    if threshold < 1.0:
        threshold = threshold * 100
    threshold = int(threshold)

    # Set values less than the threshold to zero, then
    # find genes which do not pass this threshold, i.e. have been set to zero, at all:
    Iincluded = np.zeros([Ng, L], dtype=bool)  # The genes which pass the threshold in at least atleastinconditions
    for l in range(L):
        if absvalue:
            thresholdloc = np.percentile(np.abs(Xloc[l]), threshold)
            if usereplacementval:
                Xloc[l][np.abs(Xloc[l]) < thresholdloc] = replacementVal
            Iincluded[GDMloc[:, l], l] = np.sum(np.abs(Xloc[l]) >= thresholdloc, axis=1) >= atleastinconditions
        else:
            thresholdloc = np.percentile(Xloc[l], threshold)
            if usereplacementval:
                Xloc[l][Xloc[l] < thresholdloc] = replacementVal
            Iincluded[GDMloc[:, l], l] = np.sum(Xloc[l] >= thresholdloc, axis=1) >= atleastinconditions
    Iincluded = np.sum(Iincluded, axis=1) >= atleastindatasets

    # Update Xloc, Genesloc, and finally GDM loc
    for l in range(L):
        Xloc[l] = Xloc[l][Iincluded[GDMloc[:, l]]]
    GDMloc = GDMloc[Iincluded]

    # Return results:
    return Xloc, GDMloc, Iincluded


def filterlowgenes(X, GDM, threshold=10.0, replacementVal=0.0, atleastinconditions=1, atleastindatasets=1,
                   absvalue=False, usereplacementval=False, filteringtype='raw'):
    if filteringtype == 'raw':
        return filterlowgenes_raw(X, GDM, threshold, replacementVal, atleastinconditions, atleastindatasets,
                                  absvalue, usereplacementval)
    elif filteringtype == 'perc':
        return filterlowgenes_perc(X, GDM, threshold, replacementVal, atleastinconditions, atleastindatasets,
                                   absvalue, usereplacementval)
    else:
        raise ValueError('Invalid filtering type')


def combineReplicates(X, replicatesIDs, flipSamples):
    Xloc = np.array(X)
    L = len(Xloc)

    for l in range(L):
        Xtmp = Xloc[l]
        arelogs = np.sum(abs(Xtmp) < 30) > 0.98 * ds.numel(Xtmp)  # More than 98% of values are below 30.0
        if flipSamples is not None and flipSamples[l] is not None and len(flipSamples[l]) == Xtmp.shape[1]:
            if arelogs:
                Xtmp[:, flipSamples[l] == 1] = -Xtmp[:, flipSamples[l] == 1]
            else:
                Xtmp[:, flipSamples[l] == 1] = np.divide(1.0, Xtmp[:, flipSamples[l] == 1])
        uniqueSamples = np.unique(replicatesIDs[l])
        uniqueSamples = uniqueSamples[uniqueSamples != -1]
        Xloc[l] = np.zeros([Xtmp.shape[0], len(uniqueSamples)])
        ss = 0
        for s in range(len(uniqueSamples)):
            if uniqueSamples[s] > -1:
                Xloc[l][:, ss] = np.median(Xtmp[:, replicatesIDs[l] == uniqueSamples[s]], axis=1)
                ss += 1

    return Xloc


def filterFlat(X, GDM, Iincluded):
    Xloc = np.array(X)
    GDMloc = np.array(GDM)
    Iincludedloc = np.array(Iincluded)
    L = len(Xloc)
    Ng = GDMloc.shape[0]
    Iincluded2 = np.array([False for i in range(Ng)])
    for l in range(L):
        Iincluded2[GDMloc[:, l]] = np.bitwise_or(Iincluded2[GDMloc[:,l]], np.std(Xloc[l], axis=1) > 0)

    for l in range(L):
        Xloc[l] = Xloc[l][Iincluded2[GDMloc[:,l]]]
    GDMloc = GDMloc[Iincluded2]
    Iincludedloc[Iincludedloc] = Iincluded2
    return Xloc, GDMloc, Iincludedloc


def preprocess(X, GDM, normalise=1000, replicatesIDs=None, flipSamples=None, expressionValueThreshold=10.0,
               replacementVal=0.0, atleastinconditions=1, atleastindatasets=1, absvalue=False, usereplacementval=False,
               filteringtype='raw', filterflat=True, params=None, datafiles=None):
    # Fixing parameters
    Xloc = ds.listofarrays2arrayofarrays(X)
    L = len(Xloc)
    if datafiles is None:
        if L == 1:
            datafiles = ['X']
        else:
            datafiles = np.array([], dtype=str)
            for i in range(L):
                datafiles = np.append(datafiles, 'X{0}'.format(i+1))
    if params is None:
        params = {}
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
    # Revise if the if statement below is accurate!
    if not isinstance(normalise, (list, tuple, np.ndarray)):
        normaliseloc = [normalise if isinstance(normalise, (list, tuple, np.ndarray))
                        else [normalise] for i in range(L)]
        normaliseloc = ds.listofarrays2arrayofarrays(normaliseloc)
    else:
        normaliseloc = [nor if isinstance(nor, (list, tuple, np.ndarray)) else [nor] for nor in normalise]
        normaliseloc = ds.listofarrays2arrayofarrays(normaliseloc)

    # Get rid of nans by fixing
    Xproc = Xloc
    for l in range(L):
        Xproc[l] = fixnans(Xproc[l])

    # Prepare applied_norm dictionary before any normalisation takes place
    applied_norm = collec.OrderedDict(zip(datafiles, deepcopy(normaliseloc)))

    # Tell the user if any automatic normalisation is taking place
    allare1000 = True
    anyis1000 = False
    for l in range(L):
        if 1000 in normaliseloc[l]:
            anyis1000 = True
        else:
            allare1000 = False
    if allare1000:
        io.log(' - Automatic normalisation mode (default in v1.7.0+).')
        io.log('   Clust automatically normalises your dataset(s).')
        io.log('   To switch it off, use the `-n 0` option (not recommended).')
        io.log('   Check https://github.com/BaselAbujamous/clust for details.')
    elif anyis1000:
        io.log(' - Some datasets are not assigned normalisation codes in the provided')
        io.log('   normalisation file. Clust automatically identifies and applies the')
        io.log('   most suitable normalisation to them (default in v1.7.0+).')
        io.log('   If you don''t want clust to normalise them, assign each of them a')
        io.log('   normalisation code of 0 in the normalisation file.')
        io.log('   Check https://github.com/BaselAbujamous/clust for details.')

    # Quantile normalisation
    for l in range(L):
        if 101 in normaliseloc[l] or 1000 in normaliseloc[l]:
            Xproc[l] = normaliseSampleFeatureMat(Xproc[l], 101)[0]
            if 101 in normaliseloc[l]:
                i = np.argwhere(np.array(normaliseloc[l]) == 101)
                i = i[0][0]
                normaliseloc[l][i] = 0

    # Combine replicates and sort out flipped samples
    Xproc = combineReplicates(Xproc, replicatesIDsloc, flipSamplesloc)

    # Filter genes not exceeding the threshold
    (Xproc, GDMnew, Iincluded) = filterlowgenes(Xproc, GDM, expressionValueThreshold, replacementVal,
                                                atleastinconditions, atleastindatasets, absvalue,
                                                usereplacementval, filteringtype)

    # Normalise
    for l in range(L):
        (Xproc[l], codes) = normaliseSampleFeatureMat(Xproc[l], normaliseloc[l])
        if np.all(codes == normaliseloc[l]):
            applied_norm[datafiles[l]] = op.arraytostring(applied_norm[datafiles[l]], delim=' ', openbrac='',
                                                          closebrac='')
        else:
            applied_norm[datafiles[l]] = op.arraytostring(codes, delim=' ', openbrac='', closebrac='')

    if filterflat:
        io.log(' - Flat expression profiles filtered out (default in v1.7.0+).')
        io.log('   To switch it off, use the --no-fil-flat option (not recommended).')
        io.log('   Check https://github.com/BaselAbujamous/clust for details.')
        (Xproc, GDMnew, Iincluded) = filterFlat(Xproc, GDMnew, Iincluded)

    # Prepare params for the output
    params = dict(params, **{
        'normalise': normaliseloc,
        'replicatesIDs': replicatesIDs,
        'flipSamples': flipSamplesloc,
        'L': L
    })

    return Xproc, GDMnew, Iincluded, params, applied_norm
