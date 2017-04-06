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
import sys


# Define the clust function (and below it towards the end of this file it is called).
def clustpipeline(datapath, mapfile=None, replicatesfile=None, normalisationfile=None, outpath=None,
            Ks=[n for n in range(2, 21)], tightnessweight=5, stds=0.01,
            OGsIncludedIfAtLeastInDatasets=1, expressionValueThreshold=10.0,
            atleastinconditions=1, atleastindatasets=1, smallestClusterSize=11, ncores=1, optimisation=True):
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
    glob.set_tmpfile(os.path.join(outpath, 'tmp.txt'))

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
        = pp.calculateGDMandUpdateDatasets(X, Genes, Map, mapheader=True, OGsFirstColMap=True, delimGenesInMap='W+',
                                           OGsIncludedIfAtLeastInDatasets=OGsIncludedIfAtLeastInDatasets)
    (X_summarised_normalised, GDM, Iincluded, params) = \
        pp.preprocess(X_OGs, GDM, normalise, replicatesIDs, flipSamples=None,
                      expressionValueThreshold=expressionValueThreshold, replacementVal=0.0,
                      atleastinconditions=atleastinconditions, atleastindatasets=atleastindatasets, params=None)
    OGs = OGs[Iincluded]
    if MapNew is not None:
        MapNew = MapNew[Iincluded]

    # UNCLES and M-N plots
    io.log('3. Seed clusters production (the Bi-CoPaM method)')
    ures = unc.uncles(X_summarised_normalised, type='A', GDM=GDM, Ks=Ks, params=params, Xnames=datafiles_noext,
                      ncores=ncores)
    io.log('4. Cluster evaluation and selection (the M-N scatter plots technique)')
    mnres = mn.mnplotsgreedy(X_summarised_normalised, ures.B, GDM=GDM, tightnessweight=tightnessweight,
                             params=ures.params, smallestClusterSize=smallestClusterSize, Xnames=datafiles_noext,
                             ncores=ncores)

    # Post-processing
    ppmethod = 'tukey_sqrtSCG'
    if optimisation:
        io.log('5. Cluster optimisation and completion')
        try:
            if ppmethod == 'weighted_outliers':
                B_corrected = ecorr.correcterrors_weighted_outliers(mnres.B, X_summarised_normalised, GDM,
                                                                    mnres.allDists[mnres.I], stds, smallestClusterSize)
            elif ppmethod == 'tukey_sqrtSCG':
                B_corrected = ecorr.optimise_tukey_sqrtSCG(mnres.B, X_summarised_normalised, GDM,
                                                                    mnres.allDists[mnres.I], smallestClusterSize)
            else:
                raise ValueError('Invalid post processing method (ppmethod): {0}.'.format(ppmethod))
            B_corrected = ecorr.reorderClusters(B_corrected, X_summarised_normalised, GDM)
        except:
            io.logerror(sys.exc_info())
            io.log('\n* Failed to perform cluster optimisation and completion!\n'
                   '* Skipped cluster optimisation and completion!\n')
            B_corrected = mnres.B
    else:
        io.log('5. Skipping cluster optimisation and completion')
        B_corrected = mnres.B


    # Output: Write input parameters:
    io.log('6. Saving results in\n{0}'.format(outpath))
    inputparams = op.params(mnres.params, stds, OGsIncludedIfAtLeastInDatasets,
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
    try:
        clusts_fig_file_name = '{0}/Clusters_profiles.pdf'.format(outpath)
        graph.plotclusters(X_summarised_normalised, B_corrected, clusts_fig_file_name, datafiles_noext, conditions,
                           GDM, Cs='all', setPageToDefault=True)
    except:
        io.log('Error: could not save clusters plots in a PDF file.\n'
               'Resuming producing the other results files ...')

    # Output: Prepare message to standard output and the summary then save the summary to a file and print the message
    summarymsg, endtime, timeconsumedtxt = \
        op.generateoutputsummaryparag(X, X_summarised_normalised, MapNew, GDMall, GDM,
                                      ures, mnres, B_corrected, starttime)
    summary = op.summarise_results(X, X_summarised_normalised, MapNew, GDMall, GDM,
                                   ures, mnres, B_corrected, starttime, endtime, timeconsumedtxt)
    io.writedic(outpath + '/Summary.tsv', summary, delim='\t')
    io.log(summarymsg, addextrastick=False)

    io.deletetmpfile()

