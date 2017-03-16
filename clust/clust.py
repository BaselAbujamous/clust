import matplotlib
matplotlib.use('Agg')  # To be edited as early as here before any other matplotlib import
import scripts.io as io
import scripts.uncles as unc
import scripts.mnplots as mn
import scripts.postprocess_results as ecorr
import scripts.preprocess_data as pp
import scripts.output as op
import scripts.graphics as graph
import scripts.glob as glob
import numpy as np
import os
import datetime as dt
import shutil
import argparse
from argparse import RawTextHelpFormatter


# Parse arguments
headertxt = '/==========================================================================\\\n' \
            '|                                  Clust                                   |\n' \
            '|     Optimised consensus clustering of multiple heterogenous datasets     |\n' \
            '|                               Version 1.0                                |\n' \
            '|                                                                          |\n' \
            '|                            By Basel Abu-Jamous                           |\n' \
            '|                       Department of Plant Sciences                       |\n' \
            '|                         The University of Oxford                         |\n' \
            '|                      basel.abujamous@plants.ox.ac.uk                     |\n' \
            '+--------------------------------------------------------------------------+\n' \
            '|                                 Citation                                 |\n' \
            '|                                 ~~~~~~~~                                 |\n' \
            '| When publishing work that uses Clust, please include these two citations |\n' \
            '| 1. Basel Abu-Jamous and Steve Kelly (2017) Clust (Version 1.0) [Python   |\n' \
            '|    package]. Available at https://github.com/BaselAbujamous/clust        |\n' \
            '| 2. Basel Abu-Jamous, Rui Fa, David J. Roberts, and Asoke K. Nandi (2013) |\n' \
            '|    Paradigm of tunable clustering using binarisation of consensus        |\n' \
            '|    partition matrices (Bi-CoPaM) for gene discovery, PLOS ONE, 8(2):     |\n' \
            '|    e56432.                                                               |\n' \
            '\\==========================================================================/\n'
parser = argparse.ArgumentParser(description=headertxt, formatter_class=RawTextHelpFormatter)
parser.add_argument('datapath', help='The path of the data files.')
parser.add_argument('-m', help='OGs mapping file path', default=None)
parser.add_argument('-r', help='Replicates file path', default=None)
parser.add_argument('-n', help='Normalisation file path', default=None)
parser.add_argument('-o', help='Output directory', default=None)
parser.add_argument('-K', type=int, nargs='+', help='K values (default: all values from 2 to 20 inclusively)',
                    default=[n for n in range(2, 21)])
parser.add_argument('-t', type=float, help='Cluster tightness versus cluster size weight: '
                                           'a real positive number, where 1.0 means equal weights '
                                           '(default: 1.0).', default=1.0)
parser.add_argument('-fp', type=float, help='Percentage of false positives to be trimmed, '
                                            'in the range [0.0-1.0] (default: 0.01)', default=0.01)
parser.add_argument('-d', type=int, help='Minimum number of datasets that an object has to be included in for '
                                         'it to be considered in Clust analysis. If an object is included '
                                         'only in fewer datasets than this, it will be excluded from the analysis '
                                         '(default: 1)', default=1)
parser.add_argument('-fil-v', dest='filv', type=float,
                    help='Data value (e.g. gene expression) threshold. Any value lower than this will be set to 0.0. '
                         'If an object never exceeds this value at least in -fil-c conditions in at least -fil-d '
                         'datasets, it is excluded from the analysis (default: -inf)', default=-float("inf"))
parser.add_argument('-fil-c', dest='filc', type=int,
                    help='Minimum number of conditions in a dataset in which an object should exceed the data value '
                         '-fil-v at least in -fil-d datasets to be included in the analysis (default: 0)', default=0)
parser.add_argument('-fil-d', dest='fild', type=int,
                    help='Minimum number of datasets in which an object should exceed the data value -fil-v at least '
                         'in -fil-c conditions to be included in the analysis (default: 0)', default=0)

parser.add_argument('-cs', type=int, help='Smallest cluster size (default: 11)', default=11)

args = parser.parse_args()


# Define the clust function (and below it towards the end of this file it is called).
def clust(datapath, mapfile=None, replicatesfile=None, normalisationfile=None, outpath=None,
            Ks=[n for n in range(2, 21)], tightnessweight=5, falsepositivestrimmed=0.01,
            OGsIncludedIfAtLeastInDatasets=1, expressionValueThreshold=10.0,
            atleastinconditions=1, atleastindatasets=1, smallestClusterSize=11):
    # Set the global objects label
    if mapfile is None:
        glob.set_object_label_upper('Object')
        glob.set_object_label_lower('object')
    else:
        glob.set_object_label_upper('OG')
        glob.set_object_label_lower('OG')

    # Output: Prepare the output directory and the log file
    if outpath is None:
        outpathbase = os.getcwd()
        #outpathbase = os.path.abspath(os.path.join(datapath, '..'))
        outpathbase = '{0}/Results_{1}'.format(outpathbase, dt.datetime.now().strftime('%d_%b_%y'))
        outpath = outpathbase
        trial = 0
        while os.path.exists(outpath):
            trial += 1
            outpath = '{0}_{1}'.format(outpathbase, trial)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    glob.set_logfile(os.path.join(outpath, 'log.txt'))

    # Output: Copy input files to the output
    in2out_path = outpath + '/Input_files_and_params'
    if not os.path.exists(in2out_path):
        os.makedirs(in2out_path)

    if mapfile is not None:
        shutil.copy(mapfile, os.path.join(in2out_path, 'Map.txt'))
    if replicatesfile is not None:
        shutil.copy(replicatesfile, os.path.join(in2out_path, 'Replicates.txt'))
    if normalisationfile is not None:
        shutil.copy(normalisationfile, os.path.join(in2out_path, 'Normalisation.txt'))

    in2out_X_unproc_path = in2out_path + '/Data'
    if not os.path.exists(in2out_X_unproc_path):
        os.makedirs(in2out_X_unproc_path)
    for df in io.getFilesInDirectory(datapath):
        shutil.copy(os.path.join(datapath, df), in2out_X_unproc_path)

    # Output: Print initial message, and record the starting time:
    initialmsg, starttime = op.generateinitialmessage()
    io.log(initialmsg, addextrastick=False)

    # Read data
    io.log('1. Reading datasets')
    (X, replicates, Genes, datafiles) = io.readDatasetsFromDirectory(datapath, delimiter='\t', skiprows=1, skipcolumns=1,
                                                                     returnSkipped=True)
    datafiles_noext = [os.path.splitext(d)[0] for d in datafiles]

    # Read map, replicates, and normalisation files:
    Map = io.readMap(mapfile)
    (replicatesIDs, conditions) = io.readReplicates(replicatesfile, datafiles, replicates)
    normalise = io.readNormalisation(normalisationfile, datafiles)

    # Preprocessing (Mapping then top level preprocessing including summarising replicates, filtering
    # low expression genes, and normalisation)
    io.log('2. Data pre-processing')
    (X_OGs, GDM, GDMall, OGs, MapNew, MapSpecies) \
        = pp.calculateGDMandUpdateDatasets(X, Genes, Map, mapheader=True, OGsFirstColMap=True, delimGenesInMap=';',
                                           OGsIncludedIfAtLeastInDatasets=OGsIncludedIfAtLeastInDatasets)
    (X_summarised_normalised, GDM, Iincluded, params) = \
        pp.preprocess(X_OGs, GDM, normalise, replicatesIDs, flipSamples=None,
                      expressionValueThreshold=expressionValueThreshold, replacementVal=0.0,
                      atleastinconditions=atleastinconditions, atleastindatasets=atleastindatasets, params=None)
    OGs = OGs[Iincluded]
    if MapNew is not None:
        MapNew = MapNew[Iincluded]

    # UNCLES and M-N plots
    io.log('3. Clustering (the Bi-CoPaM method)')
    ures = unc.uncles(X_summarised_normalised, type='A', GDM=GDM, Ks=Ks, params=params, Xnames=datafiles_noext)
    io.log('4. Cluster evaluation and selection (the M-N scatter plots technique)')
    mnres = mn.mnplotsgreedy(X_summarised_normalised, ures.B, GDM=GDM, tightnessweight=tightnessweight,
                             params=ures.params, smallestClusterSize=smallestClusterSize, Xnames=datafiles_noext)

    # Post-processing
    io.log('5. Error correction and cluster optimisation')
    #B_corrected = mnres.B[:, mn.mnplotsdistancethreshold(mnres.allDists[mnres.I])]
    #B_corrected = ecorr.correcterrors_withinworse(B_corrected, X_summarised_normalised, GDM, falsepositivestrimmed)

    B_corrected = ecorr.correcterrors_weighted(mnres.B, X_summarised_normalised, GDM,
                                               mnres.allDists[mnres.I], falsepositivestrimmed)
    B_corrected = ecorr.reorderClusters(B_corrected, X_summarised_normalised, GDM)


    # Output: Write input parameters:
    io.log('6. Saving results in\n{0}\n'.format(outpath))
    inputparams = op.params(mnres.params, falsepositivestrimmed, OGsIncludedIfAtLeastInDatasets,
                            expressionValueThreshold, atleastinconditions, atleastindatasets, MapNew)
    io.writedic('{0}/input_params.tsv'.format(in2out_path), inputparams, delim='\t')

    # Output: Generating and saving clusters, and processed data
    res_og = op.clusters_genes_OGs(B_corrected, OGs, MapNew, MapSpecies, '; ')
    if mapfile is None:
        np.savetxt('{0}/Clusters_Objects.tsv'.format(outpath), res_og, fmt='%s', delimiter='\t')
    else:
        np.savetxt('{0}/Clusters_OGs.tsv'.format(outpath), res_og, fmt='%s', delimiter='\t')
        res_sp = op.clusters_genes_Species(B_corrected, OGs, MapNew, MapSpecies)
        for sp in range(len(res_sp)):
            np.savetxt('{0}/Clusters_{1}.tsv'.format(outpath, MapSpecies[sp]), res_sp[sp], fmt='%s', delimiter='\t')

    Xprocessed = op.processed_X(X_summarised_normalised, conditions, GDM, OGs, MapNew, MapSpecies)
    X_proc_path = outpath + '/Processed_Data'
    if not os.path.exists(X_proc_path):
        os.makedirs(X_proc_path)
    for l in range(len(datafiles)):
        np.savetxt('{0}/{1}_processed.tsv'.format(X_proc_path, datafiles[l]), Xprocessed[l], fmt='%s', delimiter='\t')

    # Output: Save figures to a PDF
    clusts_fig_file_name = '{0}/Clusters_profiles.pdf'.format(outpath)
    graph.plotclusters(X_summarised_normalised, B_corrected, clusts_fig_file_name, datafiles_noext, conditions,
                       GDM, Cs='all', setPageToDefault=True)

    # Output: Prepare message to standard output and the summary then save the summary to a file and print the message
    summarymsg, endtime, timeconsumedtxt = \
        op.generateoutputsummaryparag(X, X_summarised_normalised, MapNew, GDMall, GDM,
                                      ures, mnres, B_corrected, starttime)
    summary = op.summarise_results(X, X_summarised_normalised, MapNew, GDMall, GDM,
                                   ures, mnres, B_corrected, starttime, endtime, timeconsumedtxt)
    io.writedic(outpath + '/Summary.tsv', summary, delim='\t')
    io.log(summarymsg, addextrastick=False)


# Call the clust function
clust(args.datapath, args.m, args.r, args.n, args.o, args.K, args.t,
        args.fp, args.d, args.filv, args.filc, args.fild, args.cs)
