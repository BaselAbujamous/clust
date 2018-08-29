import numpy as np
import datetime as dt
import collections as collec
import datastructures as ds
from glob import outputwidth, version
import pandas as pd



def msgformated(msg, alignment='<', withnewline=True):
    # alignment: '<' left, '>' right, '^' centred, and '*^' centred while padding with ******
    msgsplit = msg.split('\n')
    res = ''
    for l in msgsplit:
        while len(l) > outputwidth:
            # Cut the line at the last space or tab before the max length (outputwidth); otherwise cut at outputwith
            indexoflastspace = outputwidth - l[outputwidth-1::-1].find(' ') - 1
            indexoflasttab = outputwidth - l[outputwidth - 1::-1].find('\t') - 1
            if indexoflastspace == outputwidth and indexoflasttab == outputwidth:
                indexofcut = outputwidth  # This is the index at which cut happens
                excludeindexofcut = False  # Don't remove the character at this index as it is not a space or a tab
            elif indexoflastspace < outputwidth and indexoflasttab < outputwidth:
                indexofcut = max(indexoflastspace, indexoflasttab)
                excludeindexofcut = True
            else:
                indexofcut = min(indexoflastspace, indexoflasttab)
                excludeindexofcut = True
            res += '| {{0:{0}{1}}} |\n'.format(alignment, outputwidth).format(l[:indexofcut])
            if excludeindexofcut:
                l = l[indexofcut+1:]
            else:
                l = l[indexofcut:]
        # Add what remains of the line
        res += '| {{0:{0}{1}}} |\n'.format(alignment, outputwidth).format(l)

    if not withnewline:
        res = res[:-1]

    return res


def arraytostring(A, delim=' ', openbrac='', closebrac=''):
    L = len(A)

    if L == 0:
        res = '{0}{1}'.format(openbrac, closebrac)
    else:
        res = '{0}{1}'.format(openbrac, A[0])
        for i in range(1,L):
            res = '{0}{1}{2}'.format(res, delim, A[i])
        res = '{0}{1}'.format(res, closebrac)

    return res


def topline(withnewline=True):
    res = '/{{0:=^{0}}}\\'.format(outputwidth+2).format('')
    if withnewline:
        res += '\n'
    return res


def bottomline(withnewline=True):
    res = '\\{{0:=^{0}}}/'.format(outputwidth+2).format('')
    if withnewline:
        res += '\n'
    return res


def midline(withnewline=True):
    res = '+{{0:-^{0}}}+'.format(outputwidth+2).format('')
    if withnewline:
        res += '\n'
    return res


def generateinitialmessage():
    starttime = dt.datetime.now()

    res = '\n' + topline()
    tmptxt = 'Clust\n' \
             '(Optimised consensus clustering of multiple heterogenous datasets)\n' \
             'Python package version {0} (2018) Basel Abu-Jamous'.format(version[1:])
    res += msgformated(tmptxt, alignment='^')
    res += midline()

    tmptxt = 'Analysis started at: {0}'.format(starttime.strftime('%A %d %B %Y (%H:%M:%S)'))
    res += msgformated(tmptxt, withnewline=False)
    return res, starttime


def generateoutputsummaryparag(X, Xprocessed, Map, GDMall, GDM, uncle_res, mn_res, B_corrected, starttime):
    # Sort out the ending time and consumed time
    endtime = dt.datetime.now()
    t = endtime - starttime
    d = t.days
    h = t.seconds / 3600
    m = (t.seconds % 3600) / 60
    s = t.seconds % 60
    if t.total_seconds() < 1:
        mil = t.microseconds / 1000
        timeConsumedTxt = '{0} millisecond{1}'.format(mil, '' if mil == 1 else 's')
    else:
        timeConsumedTxt = '{0}{1}{2} {3}, {4} {5}, and {6} {7}' \
                          ''.format(d if d > 0 else '', ' days, ' if d > 1 else ' day, ' if d == 1 else '',
                                    h, 'hour' if h == 1 else 'hours',
                                    m, 'minute' if m == 1 else 'minutes',
                                    s, 'second' if s == 1 else 'seconds')

    # Generate the text
    L = len(X)
    label0 = 'genes' if Map is None else 'OrthoGroups (OGs)*'
    label1 = 'gene' if Map is None else 'OG'
    dataset_datasets = 'dataset' if L == 1 else 'datasets'
    Ng_original = GDMall.shape[0]
    Ng_to_clustering = GDM.shape[0]
    Ng_in_clusters = np.sum(np.any(B_corrected, axis=1))

    tmptxt = 'Analysis finished at: {0}\n' \
             'Total time consumed: {1}\n'.format(endtime.strftime('%A %d %B %Y (%H:%M:%S)'), timeConsumedTxt)
    res = midline()
    res += msgformated(tmptxt)
    res += bottomline()
    res += '\n' + topline()
    res += msgformated('RESULTS SUMMARY', alignment='^')
    res += midline()

    K = B_corrected.shape[1]

    if K > 0:
        tmptxt = 'Clust received {2} {3} with {4} unique {0}. After filtering, {5} {1}s made it to the clustering ' \
                 'step. Clust generated {6} clusters of {1}s, which in total include {7} {1}s. The smallest cluster ' \
                 'includes {8} {1}s, the largest cluster includes {9} {1}s, and the average cluster size ' \
                 'is {10} {1}s.' \
                 ''.format(label0, label1, L, dataset_datasets, Ng_original, Ng_to_clustering, K, Ng_in_clusters,
                           np.min(np.sum(B_corrected, axis=0)), np.max(np.sum(B_corrected, axis=0)),
                           np.mean(np.sum(B_corrected, axis=0)))
    else:
        tmptxt = 'Clust received {2} {3} with {4} unique {0}. After filtering, {5} {1}s made it to the clustering ' \
                 'step. Clust generated {6} clusters of {1}s.' \
                 ''.format(label0, label1, L, dataset_datasets, Ng_original, Ng_to_clustering, K)
    res += msgformated(tmptxt)
    if Map is not None:
        tmptxt = '\n* An OG is a group of orthologous genes within & across different species, as identified by the ' \
                 'provided map.'
        res += msgformated(tmptxt)

    res += midline()

    res += msgformated('Citation\n~~~~~~~~', alignment='^')
    tmptxt = 'When publishing work that uses Clust, please include this citation:\n' \
             'Basel Abu-Jamous and Steven Kelly (2018) Clust: automatic extraction of optimal co-expressed ' \
             'gene clusters from gene expression data. bioRxiv 221309; doi: https://doi.org/10.1101/221309.'

    res += msgformated(tmptxt)
    res += midline()

    tmptxt = 'For enquiries contact:\n' \
             'Basel Abu-Jamous\n' \
             'Department of Plant Sciences, University of Oxford\n' \
             'basel.abujamous@plants.ox.ac.uk\n' \
             'baselabujamous@gmail.com'
    res += msgformated(tmptxt)
    res += bottomline(withnewline=False)

    return res, endtime, timeConsumedTxt


def summarise_results(X, Xprocessed, Map, GDMall, GDM, uncle_res, mn_res,
                      B_corrected, starttime, endtime, timeconsumedtxt):
    K = B_corrected.shape[1]
    if K > 0:
        if Map is None:
            if len(X) == 1:
                res = collec.OrderedDict(
                    [('Starting data and time', starttime.strftime('%A %d %B %Y (%H:%M:%S)')),
                     ('Ending date and time', endtime.strftime('%A %d %B %Y (%H:%M:%S)')),
                     ('Time consumed', timeconsumedtxt),
                     ('Number of datasets', len(X)),
                     ('Total number of input genes', GDMall.shape[0]),
                     ('Genes included in the analysis', GDM.shape[0]),
                     ('Genes filtered out from the analysis', GDMall.shape[0] - GDM.shape[0]),
                     ('Number of clusters', B_corrected.shape[1]),
                     ('Total number of genes in clusters', np.sum(np.any(B_corrected, axis=1))),
                     ('Genes not included in any cluster', B_corrected.shape[0] - np.sum(np.any(B_corrected, axis=1))),
                     ('Min cluster size', np.min(np.sum(B_corrected, axis=0))),
                     ('Max cluster size', np.max(np.sum(B_corrected, axis=0))),
                     ('Average cluster size', np.mean(np.sum(B_corrected, axis=0))),
                     ('Number of samples (including replicates)', np.shape(X[0])[1]),
                     ('Number of conditions', np.shape(Xprocessed[0])[1]),
                     ])
            else:
                res = collec.OrderedDict(
                    [('Starting data and time', starttime.strftime('%A %d %B %Y (%H:%M:%S)')),
                     ('Ending date and time', endtime.strftime('%A %d %B %Y (%H:%M:%S)')),
                     ('Time consumed', timeconsumedtxt),
                     ('Number of datasets', len(X)),
                     ('Total number of input genes', GDMall.shape[0]),
                     ('Genes included in the analysis', GDM.shape[0]),
                     ('Genes filtered out from the analysis', GDMall.shape[0] - GDM.shape[0]),
                     ('Number of clusters', B_corrected.shape[1]),
                     ('Total number of genes in clusters', np.sum(np.any(B_corrected, axis=1))),
                     ('Genes not included in any cluster', B_corrected.shape[0] - np.sum(np.any(B_corrected, axis=1))),
                     ('Min cluster size', np.min(np.sum(B_corrected, axis=0))),
                     ('Max cluster size', np.max(np.sum(B_corrected, axis=0))),
                     ('Average cluster size', np.mean(np.sum(B_corrected, axis=0))),
                     ('Total number of samples (including replicates)', np.sum([np.shape(x)[1] for x in X])),
                     ('Average number of samples (including replicates) per dataset', np.mean([np.shape(x)[1] for x in X])),
                     ('Total number of conditions', np.sum([np.shape(x)[1] for x in Xprocessed])),
                     ('Average number of conditions per dataset', np.mean([np.shape(x)[1] for x in Xprocessed])),
                     ('nokey0', '**********************************'),
                     ('Genes included in all datasets', np.sum(np.all(GDMall, axis=1))),
                     ('Genes missed from 1 dataset', np.sum(np.sum(GDMall == 0, axis=1) == 1)),
                     ('Genes missed from 2 datasets', np.sum(np.sum(GDMall == 0, axis=1) == 2)),
                     ('Genes missed from 3 datasets', np.sum(np.sum(GDMall == 0, axis=1) == 3)),
                     ('Genes missed from more than 3 datasets', np.sum(np.sum(GDMall == 0, axis=1) > 3))
                     ])
        else:
            genesOGs = np.array([[len(og_species) for og_species in og] for og in Map]) # Number of genes per OG per species
            res = collec.OrderedDict(
                [('Starting data and time', starttime.strftime('%A %d %B %Y (%H:%M:%S)')),
                 ('Ending date and time', endtime.strftime('%A %d %B %Y (%H:%M:%S)')),
                 ('Time consumed', timeconsumedtxt),
                 ('Number of datasets', len(X)),
                 ('Total number of input OGs', GDMall.shape[0]),
                 ('OGs included in the analysis', GDM.shape[0]),
                 ('OGs filtered out from the analysis', GDMall.shape[0] - GDM.shape[0]),
                 ('Number of clusters', B_corrected.shape[1]),
                 ('Total number of OGs in clusters', np.sum(np.any(B_corrected, axis=1))),
                 ('OGs not included in any cluster', B_corrected.shape[0] - np.sum(np.any(B_corrected, axis=1))),
                 ('Min cluster size', np.min(np.sum(B_corrected, axis=0))),
                 ('Max cluster size', np.max(np.sum(B_corrected, axis=0))),
                 ('Average cluster size', np.mean(np.sum(B_corrected, axis=0))),
                 ('Total number of samples (including replicates)', np.sum([np.shape(x)[1] for x in X])),
                 ('Average number of samples (including replicates) per dataset', np.mean([np.shape(x)[1] for x in X])),
                 ('Total number of conditions', np.sum([np.shape(x)[1] for x in Xprocessed])),
                 ('Average number of conditions per dataset', np.mean([np.shape(x)[1] for x in Xprocessed])),
                 ('nokey0', '**********************************'),
                 ('Number of species', Map.shape[1]),
                 ('Average number of genes in an OG per type', np.mean(genesOGs)),
                 ('Max number of genes in an OG per type', np.max(genesOGs)),
                 ('Number of OGs with at least a single gene in each type', np.sum(np.all(genesOGs >= 1, axis=1))),
                 ('Number of OGs with exactly a single gene in each type', np.sum(np.all(genesOGs == 1, axis=1))),
                 ('nokey1', '**********************************'),
                 ('OGs included in all datasets', np.sum(np.all(GDMall, axis=1))),
                 ('OGs missed from 1 dataset', np.sum(np.sum(GDMall == 0, axis=1) == 1)),
                 ('OGs missed from 2 datasets', np.sum(np.sum(GDMall == 0, axis=1) == 2)),
                 ('OGs missed from 3 datasets', np.sum(np.sum(GDMall == 0, axis=1) == 3)),
                 ('OGs missed from more than 3 datasets', np.sum(np.sum(GDMall == 0, axis=1) > 3))
                 ])
    else:
        if Map is None:
            res = collec.OrderedDict(
                [('Starting data and time', starttime.strftime('%A %d %B %Y (%H:%M:%S)')),
                 ('Ending date and time', endtime.strftime('%A %d %B %Y (%H:%M:%S)')),
                 ('Time consumed', timeconsumedtxt),
                 ('Number of datasets', len(X)),
                 ('Total number of input genes', GDMall.shape[0]),
                 ('Genes included in the analysis', GDM.shape[0]),
                 ('Genes filtered out from the analysis', GDMall.shape[0] - GDM.shape[0]),
                 ('Number of clusters', B_corrected.shape[1]),
                 ('nokey0', '**********************************'),
                 ('Genes included in all datasets', np.sum(np.all(GDMall, axis=1))),
                 ('Genes missed from 1 dataset', np.sum(np.sum(GDMall == 0, axis=1) == 1)),
                 ('Genes missed from 2 datasets', np.sum(np.sum(GDMall == 0, axis=1) == 2)),
                 ('Genes missed from 3 datasets', np.sum(np.sum(GDMall == 0, axis=1) == 3)),
                 ('Genes missed from more than 3 datasets', np.sum(np.sum(GDMall == 0, axis=1) > 3))
                 ])
        else:
            genesOGs = np.array([[len(og_species) for og_species in og] for og in Map]) # Number of genes per OG per species
            res = collec.OrderedDict(
                [('Starting data and time', starttime.strftime('%A %d %B %Y (%H:%M:%S)')),
                 ('Ending date and time', endtime.strftime('%A %d %B %Y (%H:%M:%S)')),
                 ('Time consumed', timeconsumedtxt),
                 ('Number of datasets', len(X)),
                 ('Total number of input OGs', GDMall.shape[0]),
                 ('OGs included in the analysis', GDM.shape[0]),
                 ('OGs filtered out from the analysis', GDMall.shape[0] - GDM.shape[0]),
                 ('Number of clusters', B_corrected.shape[1]),
                 ('nokey0', '**********************************'),
                 ('Number of species', Map.shape[1]),
                 ('Average number of genes in an OG per type', np.mean(genesOGs)),
                 ('Max number of genes in an OG per type', np.max(genesOGs)),
                 ('Number of OGs with at least a single gene in each type', np.sum(np.all(genesOGs >= 1, axis=1))),
                 ('Number of OGs with exactly a single gene in each type', np.sum(np.all(genesOGs == 1, axis=1))),
                 ('nokey1', '**********************************'),
                 ('OGs included in all datasets', np.sum(np.all(GDMall, axis=1))),
                 ('OGs missed from 1 dataset', np.sum(np.sum(GDMall == 0, axis=1) == 1)),
                 ('OGs missed from 2 datasets', np.sum(np.sum(GDMall == 0, axis=1) == 2)),
                 ('OGs missed from 3 datasets', np.sum(np.sum(GDMall == 0, axis=1) == 3)),
                 ('OGs missed from more than 3 datasets', np.sum(np.sum(GDMall == 0, axis=1) > 3))
                 ])
    return res


def clusters_genes_OGs(B, OGs, Map, MapSpecies, delim='; '):
    if Map is None:
        Nsp = 0
    else:
        Nsp = len(MapSpecies)  # Number of species
    K = B.shape[1]  # Number of clusters
    if K == 0:
        return np.array(np.empty([1, 1]), dtype=object)
    Csizes = np.sum(B, axis=0)  # Clusters' sizes
    maxCsize = np.max(Csizes)  # Largest cluster size
    res = np.array(np.empty([maxCsize, (Nsp + 1) * K], dtype=str), dtype=object)
    header = np.array([None] * ((Nsp + 1) * K * 2), dtype=object).reshape([2, ((Nsp + 1) * K)])
    for k in range(K):
        col = k * (Nsp + 1)
        res[0:Csizes[k], col] = OGs[B[:, k]]
        res[Csizes[k]:, col] = ''
        header[0, col] = 'C{0} ({1} {2})'.format(k, Csizes[k], 'genes' if Map is None else 'OGs')
        header[1, col] = 'Genes' if Map is None else 'OGs'
        for sp in range(Nsp):  # Will not get into this if Map is None, as Nsp will be zero in that case
            col = k * (Nsp + 1) + sp + 1
            res[0:Csizes[k], col] = [ds.concatenateStrings(gs, delim) for gs in Map[B[:, k], sp]]
            res[Csizes[k]:, col] = ''
            header[0, col] = ''
            header[1, col] = MapSpecies[sp]


    return pd.DataFrame(data=np.array(np.concatenate((header, res), axis=0), dtype=str), index=None, columns=None, dtype=str)


def clusters_genes_Species(B, OGs, Map, MapSpecies):
    Nsp = len(MapSpecies)  # Number of species
    K = B.shape[1]  # Number of clusters

    # Find flattened genes in species
    flatGenesInSpecies = [[ds.flattenAList(Map[B[:, k], sp]) for k in range(K)] for sp in range(Nsp)]  # Nsp x K lists

    # Prepare the results object
    Csizes = [[len(sp_k_genes) for sp_k_genes in sp_genes] for sp_genes in flatGenesInSpecies]  # Nsp x K
    maxCsizes = [np.max(csizes_sp) for csizes_sp in Csizes]  # Nsp x 1
    res = np.array([None] * Nsp, dtype=object)
    resFrames = np.array([None] * Nsp, dtype=object)

    # Fill the results object, species by species
    for sp in range(Nsp):
        restmp = np.array(np.empty([maxCsizes[sp], K], dtype=str), dtype=object)
        header = np.array([None] * K, dtype=object).reshape([1, K])
        for k in range(K):
            restmp[0:Csizes[sp][k], k] = flatGenesInSpecies[sp][k]
            restmp[Csizes[sp][k]:, k] = ''
            header[0, k] = 'C{0} ({1} genes)'.format(k, Csizes[sp][k])
        res[sp] = np.array(np.concatenate((header, restmp), axis=0), dtype=str)
        resFrames[sp] = pd.DataFrame(data=res[sp], columns=None, index=None, dtype=str)

    return resFrames


def processed_X(Xprocessed, conditions, GDM, OGs, Map, MapSpecies):
    L = len(Xprocessed)
    res = np.array([None] * L, dtype=object)
    resFrame = np.array([None] * L, dtype=object)
    resData = np.array([None] * L, dtype=object)
    resHeader = np.array([None] * L, dtype=object)
    resGeneNames = np.array([None] * L, dtype=object)

    for l in range(L):
        # Header (Samples)
        #samploc = np.array(Samples[l])
        #uniqueSamploc = np.unique(SamplesIDs[l])
        #uniqueSamploc = uniqueSamploc[uniqueSamploc >= 0]
        #resHeader[l] = [samploc[np.array(SamplesIDs[l]) == s][0] for s in uniqueSamploc]
        resHeader[l] = conditions[l]
        if Map is None:
            resHeader[l] = np.array([['Genes'] + resHeader[l]])
        else:
            resHeader[l] = np.array([['OGs'] + MapSpecies.tolist() + resHeader[l]])

        # Gene names
        if Map is None:
            resGeneNames[l] = OGs[GDM[:, l]].reshape(-1,1)
        else:
            genenames = [[ds.concatenateStrings(gs) for gs in Map[GDM[:, l]][:, sp]] for sp in range(Map.shape[1])]
            resGeneNames[l] = np.concatenate((OGs[GDM[:, l]].reshape(-1,1), np.transpose(genenames)), axis=1)

        # Data
        resData[l] = np.array(Xprocessed[l])

        # concatenate them
        res[l] = np.concatenate((resGeneNames[l], resData[l]), axis=1)
        res[l] = np.concatenate((resHeader[l], res[l]), axis=0)
        res[l] = np.array(res[l], dtype=str)

        resFrame[l] = pd.DataFrame(data=res[l], index=None, columns=None, dtype=str)

    return resFrame


def params(params, q3s, OGsIncludedIfAtLeastInDatasets, expressionValueThreshold,
           atleastinconditions, atleastindatasets, deterministic, methods, Map=None):
    if Map is None:
        res = collec.OrderedDict(
            [('Methods', params['methods']),
             ('K values (Ks)', params['Ks']),
             ('Tightness weight', params['tightnessweight']),
             ('Q3s outliers threshold', q3s),
             ('Genes included if at least in datasets', OGsIncludedIfAtLeastInDatasets),
             ('Filtering: Data value threshold', expressionValueThreshold),
             ('Filtering: At least in conditions', atleastinconditions),
             ('Filtering: At least in datasets', atleastindatasets),
             ('Is deterministic?', deterministic),
             ('Seed methods', methods)
             ])
    else:
        res = collec.OrderedDict(
            [('Methods', params['methods']),
             ('K values (Ks)', params['Ks']),
             ('Tightness weight', params['tightnessweight']),
             ('Q3s outliers threshold', q3s),
             ('OGs included if at least in datasets', OGsIncludedIfAtLeastInDatasets),
             ('Filtering: Data value threshold', expressionValueThreshold),
             ('Filtering: At least in conditions', atleastinconditions),
             ('Filtering: At least in datasets', atleastindatasets),
             ('Is deterministic?', deterministic),
             ('Seed methods', methods)
             ])
    return res
