import matplotlib
matplotlib.use('Agg')  # To be edited as early as here before any other matplotlib import
import clust.scripts.io as io
import clust.scripts.uncles as unc
import clust.scripts.mnplots as mn
import clust.scripts.postprocess_results as ecorr
import clust.scripts.preprocess_data as pp
import clust.scripts.output as op
import clust.scripts.graphics as graph
import clust.scripts.glob as glob
import clust.scripts.numeric as nu
import clust.scripts.eigengene as eig
import clust.scripts.validation as val
import clust.scripts.datastructures as ds
import numpy as np
import os
import datetime as dt
import shutil
import sys
import pandas as pd
import math


# Define the clust function (and below it towards the end of this file it is called).
def clustpipeline(datapath, mapfile=None, replicatesfile=None, normalisationfile=['1000'], outpath=None,
                  Ks=[n for n in range(4, 21, 4)], tightnessweight=1, stds=3.0,
                  OGsIncludedIfAtLeastInDatasets=1, expressionValueThreshold=-float("inf"), atleastinconditions=0,
                  atleastindatasets=0, absvalue=False, filteringtype='raw', filflat=True, smallestClusterSize=11,
                  ncores=1, optimisation=True, Q3s=2, methods=None, deterministic=False):
    # Set the global objects label
    if mapfile is None:
        glob.set_object_label_upper('Gene')
        glob.set_object_label_lower('gene')
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
        if len(normalisationfile) == 1 and not nu.isint(normalisationfile[0]):
            shutil.copy(normalisationfile[0], os.path.join(in2out_path, 'Normalisation.txt'))

    in2out_X_unproc_path = in2out_path + '/Data'
    if not os.path.exists(in2out_X_unproc_path):
        os.makedirs(in2out_X_unproc_path)
    if os.path.isfile(datapath):
        shutil.copy(datapath, in2out_X_unproc_path)
    elif os.path.isdir(datapath):
        for df in io.getFilesInDirectory(datapath):
            shutil.copy(os.path.join(datapath, df), in2out_X_unproc_path)
    else:
        raise ValueError('Data path {0} does not exist. Either provide a path '.format(datapath) + \
                         'of a data file or a path to a directory including data file(s)')


    # Output: Print initial message, and record the starting time:
    initialmsg, starttime = op.generateinitialmessage()
    io.log(initialmsg, addextrastick=False)

    # Read data
    io.log('1. Reading dataset(s)')
    (X, replicates, Genes, datafiles) = io.readDatasetsFromDirectory(datapath, delimiter='\t| |, |; |,|;', skiprows=1, skipcolumns=1,
                                                                     returnSkipped=True)
    datafiles_noext = [os.path.splitext(d)[0] for d in datafiles]

    # Read map, replicates, and normalisation files:
    Map = io.readMap(mapfile)
    (replicatesIDs, conditions) = io.readReplicates(replicatesfile, datapath, datafiles, replicates)
    normalise = io.readNormalisation(normalisationfile, datafiles)

    # Preprocessing (Mapping then top level preprocessing including summarising replicates, filtering
    # low expression genes, and normalisation)
    io.log('2. Data pre-processing')
    (X_OGs, GDM, GDMall, OGs, MapNew, MapSpecies) \
        = pp.calculateGDMandUpdateDatasets(X, Genes, Map, mapheader=True, OGsFirstColMap=True, delimGenesInMap='\\W+',
                                           OGsIncludedIfAtLeastInDatasets=OGsIncludedIfAtLeastInDatasets)
    (X_summarised_normalised, GDM, Iincluded, params, applied_norms) = \
        pp.preprocess(X_OGs, GDM, normalise, replicatesIDs, flipSamples=None,
                      expressionValueThreshold=expressionValueThreshold, replacementVal=0.0,
                      atleastinconditions=atleastinconditions, atleastindatasets=atleastindatasets, absvalue=absvalue,
                      filteringtype=filteringtype, filterflat=filflat, params=None, datafiles=datafiles)
    io.writedic('{0}/Normalisation_actual.txt'.format(outpath), applied_norms, delim='\t')
    OGs = OGs[Iincluded]
    if MapNew is not None:
        MapNew = MapNew[Iincluded]

    # Output: Save processed data
    Xprocessed = op.processed_X(X_summarised_normalised, conditions, GDM, OGs, MapNew, MapSpecies)  # pandas DataFrames
    X_proc_path = outpath + '/Processed_Data'
    if not os.path.exists(X_proc_path):
        os.makedirs(X_proc_path)
    for l in range(len(datafiles)):
        pd.DataFrame.to_csv(Xprocessed[l], '{0}/{1}_processed.tsv'.format(X_proc_path, datafiles[l]), sep='\t', encoding='utf-8', index=None, columns=None, header=False)
        #np.savetxt('{0}/{1}_processed.tsv'.format(X_proc_path, datafiles[l]), Xprocessed[l], fmt='%s', delimiter='\t')


    # UNCLES and M-N plots
    io.log('3. Seed clusters production (the Bi-CoPaM method)')
    ures = unc.uncles(X_summarised_normalised, type='A', GDM=GDM, Ks=Ks, params=params, methods=methods,
                      Xnames=datafiles_noext, ncores=ncores, deterministic=deterministic)
    io.log('4. Cluster evaluation and selection (the M-N scatter plots technique)')
    mnres = mn.mnplotsgreedy(X_summarised_normalised, ures.B, GDM=GDM, tightnessweight=tightnessweight,
                             params=ures.params, smallestClusterSize=smallestClusterSize, Xnames=datafiles_noext,
                             ncores=ncores)

    # Post-processing
    ppmethod = 'tukey_sqrtSCG'
    if optimisation:
        io.log('5. Cluster optimisation and completion')
        if len(mnres.I) > 0 and sum(mnres.I) > 0:  # Otherwise, there are no clusters, so nothing to be corrected
            try:
                if ppmethod == 'weighted_outliers':
                    B_corrected = ecorr.correcterrors_weighted_outliers(mnres.B, X_summarised_normalised, GDM,
                                                                        mnres.allDists[mnres.I], stds, smallestClusterSize)
                elif ppmethod == 'tukey_sqrtSCG':
                    B_corrected = ecorr.optimise_tukey_sqrtSCG(mnres.B, X_summarised_normalised, GDM,
                                                                        mnres.allDists[mnres.I], smallestClusterSize,
                                                               tails=1, Q3s=Q3s)
                else:
                    raise ValueError('Invalid post processing method (ppmethod): {0}.'.format(ppmethod))
                B_corrected = ecorr.reorderClusters(B_corrected, X_summarised_normalised, GDM)
            except:
                io.logerror(sys.exc_info())
                io.log('\n* Failed to perform cluster optimisation and completion!\n'
                       '* Skipped cluster optimisation and completion!\n')
                B_corrected = mnres.B
        else:
            B_corrected = mnres.B
    else:
        io.log('5. Skipping cluster optimisation and completion')
        B_corrected = mnres.B


    # Output: Write input parameters:
    io.log('6. Saving results in\n{0}'.format(outpath))
    inputparams = op.params(mnres.params, Q3s, OGsIncludedIfAtLeastInDatasets,
                            expressionValueThreshold, atleastinconditions, atleastindatasets,
                            deterministic, ures.params['methods'], MapNew)
    io.writedic('{0}/input_params.tsv'.format(in2out_path), inputparams, delim='\t')

    # Output: Generating and saving clusters
    res_og = op.clusters_genes_OGs(B_corrected, OGs, MapNew, MapSpecies, '; ')  # pandas DataFrame
    if mapfile is None:
        pd.DataFrame.to_csv(res_og, '{0}/Clusters_Objects.tsv'.format(outpath), sep='\t',
                            encoding='utf-8', index=None, columns=None, header=False)
        #np.savetxt('{0}/Clusters_Objects.tsv'.format(outpath), res_og, fmt='%s', delimiter='\t')
    else:
        pd.DataFrame.to_csv(res_og, '{0}/Clusters_OGs.tsv'.format(outpath), sep='\t',
                            encoding='utf-8', index=None, columns=None, header=False)
        #np.savetxt('{0}/Clusters_OGs.tsv'.format(outpath), res_og, fmt='%s', delimiter='\t')
        res_sp = op.clusters_genes_Species(B_corrected, OGs, MapNew, MapSpecies)  # pandas DataFrame
        for sp in range(len(res_sp)):
            pd.DataFrame.to_csv(res_sp[sp], '{0}/Clusters_{1}.tsv'.format(outpath, MapSpecies[sp]), sep='\t',
                                encoding='utf-8', index=None, columns=None, header=False)
            #np.savetxt('{0}/Clusters_{1}.tsv'.format(outpath, MapSpecies[sp]), res_sp[sp], fmt='%s', delimiter='\t')

    # Output: Save figures to a PDF

    try:
        if np.shape(B_corrected)[1] > 0:
            clusts_fig_file_name = '{0}/Clusters_profiles.pdf'.format(outpath)
            graph.plotclusters(X_summarised_normalised, B_corrected, datafiles_noext, conditions, clusts_fig_file_name,
                               GDM=GDM, Cs='all', setPageToDefault=True, printToPDF=True, showPlots=False)
    except:
        io.log('Error: could not save clusters plots in a PDF file.\n'
               'Resuming producing the other results files ...')

    # Output: Generating and writing eigengenes
    try:
        if np.shape(B_corrected)[1] > 0:
            if len(X_summarised_normalised) == 1:
                eigengene_matrix = eig.eigengenes_dataframe(X_summarised_normalised, B_corrected, conditions)
                eigengene_matrix.to_csv('{0}/Eigengenes.tsv'.format(outpath), sep='\t',
                                    encoding='utf-8')
            else:
                io.log('Eigengene computation is currently not supported for multiple datasets.')
    except:
        io.log('Error: could not save eigengenes into a file.\n'
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

def runclust(X, Map=None, replicatesIDs=None, normalise=1000,
             Ks=[n for n in range(4, 21, 4)], tightnessweight=1, stds=3.0,
             OGsIncludedIfAtLeastInDatasets=1, expressionValueThreshold=-float("inf"), atleastinconditions=0,
             atleastindatasets=0, absvalue=False, filteringtype='raw', filflat=True, smallestClusterSize=11,
             ncores=1, optimisation=True, Q3s=2, methods=None, deterministic=False, showPlots=True,
             printToConsole=True):

    # Set the global objects label
    glob.set_print_to_log_file(False)
    glob.set_print_to_console(printToConsole)
    if Map is None:
        glob.set_object_label_upper('Gene')
        glob.set_object_label_lower('gene')
    else:
        glob.set_object_label_upper('OG')
        glob.set_object_label_lower('OG')

    glob.set_tmpfile('clust_tmp.txt')


    # Output: Print initial message, and record the starting time:
    initialmsg, starttime = op.generateinitialmessage()
    io.log(initialmsg, addextrastick=False)

    # Consider X as a list of arrays or of data frames. Otherwise, make it as such first
    # If the user entered a single dataset as an input (not as a list of arrays), save this fact in a flag, ...
    # so the result is returned as a single output
    input_is_one_dataset = False
    if isinstance(X, pd.DataFrame):
        input_is_one_dataset = True
        X = [X]
    elif isinstance(X, np.ndarray) and ds.maxDepthOfArray(X) == 2:
        input_is_one_dataset = True
        X = [X]

    # Format data (X: list of arrays, Genes: list of arrays of strings, replicates: list of arrays of strings)
    L = len(X)  # Number of datasets
    replicates = [None] * L
    Genes = [None] * L
    io.log('1. Reading dataset(s)')
    for l in range(L):
        if type(X[l]) == pd.DataFrame:
            Genes[l] = np.array(X[l].index, dtype=str, ndmin=2).transpose()
            Genes[l] = np.array(Genes[l], dtype=object)
            replicates[l] = np.array(X[l].columns, dtype=str)
            X[l] = X[l].values
        else:
            X[l] = np.array(X[l])

            ngenes_digits = int(math.ceil(math.log10(X[l].shape[0])))
            nreps_digits = int(math.ceil(math.log10(X[l].shape[1])))
            Genes[l] = np.array([['{0}'.format(str(g).zfill(ngenes_digits))] for g in range(X[l].shape[0])])
            Genes[l] = np.array(Genes[l], dtype=object)
            replicates[l] = np.array(['X{0}'.format(str(r).zfill(nreps_digits)) for r in range(X[l].shape[1])])

    ndatasets_digits = int(math.ceil(math.log10(L)))
    datafiles = np.array(['D{0}'.format(str(r).zfill(ndatasets_digits)) for r in range(L)])
    datafiles_noext = datafiles

    # Sort out conditions based on replicates structure if given
    if replicatesIDs is None:
        conditions = replicates
    else:
        valresult = val.validate_replicatesIDs(replicatesIDs, X)
        if valresult[0]:
            conditions = [None] * L
            for l in range(L):
                if replicatesIDs[l] is None:
                    conditions[l] = np.array(replicates[l])
                else:
                    uniq_reps, cond_indices = np.unique(replicatesIDs[l], return_index=True)
                    if -1 in uniq_reps:
                        cond_indices = cond_indices[1:]
                    conditions[l] = replicates[l][cond_indices]
        else:
            io.log(valresult[1])
            io.log("Terminating ...")
            raise Exception("Terminated by an invalid input argument.")

    # Validate normalisation
    valresult = val.validate_normalisation(normalise, X)
    if not valresult[0]:
        io.log(valresult[1])
        io.log("Terminating ...")
        raise Exception("Terminated by an invalid input argument.")


    # Preprocessing (Mapping then top level preprocessing including summarising replicates, filtering
    # low expression genes, and normalisation)
    io.log('2. Data pre-processing')
    (X_OGs, GDM, GDMall, OGs, MapNew, MapSpecies) \
        = pp.calculateGDMandUpdateDatasets(X, Genes, Map, mapheader=True, OGsFirstColMap=True, delimGenesInMap='\\W+',
                                           OGsIncludedIfAtLeastInDatasets=OGsIncludedIfAtLeastInDatasets)
    (X_summarised_normalised, GDM, Iincluded, params, applied_norms) = \
        pp.preprocess(X_OGs, GDM, normalise, replicatesIDs, flipSamples=None,
                      expressionValueThreshold=expressionValueThreshold, replacementVal=0.0,
                      atleastinconditions=atleastinconditions, atleastindatasets=atleastindatasets, absvalue=absvalue,
                      filteringtype=filteringtype, filterflat=filflat, params=None, datafiles=datafiles)
    OGs = OGs[Iincluded]
    if MapNew is not None:
        MapNew = MapNew[Iincluded]

    # UNCLES and M-N plots
    io.log('3. Seed clusters production (the Bi-CoPaM method)')
    ures = unc.uncles(X_summarised_normalised, type='A', GDM=GDM, Ks=Ks, params=params, methods=methods,
                      Xnames=datafiles_noext, ncores=ncores, deterministic=deterministic)
    io.log('4. Cluster evaluation and selection (the M-N scatter plots technique)')
    mnres = mn.mnplotsgreedy(X_summarised_normalised, ures.B, GDM=GDM, tightnessweight=tightnessweight,
                             params=ures.params, smallestClusterSize=smallestClusterSize, Xnames=datafiles_noext,
                             ncores=ncores)

    # Post-processing
    ppmethod = 'tukey_sqrtSCG'
    if optimisation:
        io.log('5. Cluster optimisation and completion')
        if len(mnres.I) > 0 and sum(mnres.I) > 0:  # Otherwise, there are no clusters, so nothing to be corrected
            try:
                if ppmethod == 'weighted_outliers':
                    B_corrected = ecorr.correcterrors_weighted_outliers(mnres.B, X_summarised_normalised, GDM,
                                                                        mnres.allDists[mnres.I], stds, smallestClusterSize)
                elif ppmethod == 'tukey_sqrtSCG':
                    B_corrected = ecorr.optimise_tukey_sqrtSCG(mnres.B, X_summarised_normalised, GDM,
                                                                        mnres.allDists[mnres.I], smallestClusterSize,
                                                               tails=1, Q3s=Q3s)
                else:
                    raise ValueError('Invalid post processing method (ppmethod): {0}.'.format(ppmethod))
                B_corrected = ecorr.reorderClusters(B_corrected, X_summarised_normalised, GDM)
            except:
                io.logerror(sys.exc_info())
                io.log('\n* Failed to perform cluster optimisation and completion!\n'
                       '* Skipped cluster optimisation and completion!\n')
                B_corrected = mnres.B
        else:
            B_corrected = mnres.B
    else:
        io.log('5. Skipping cluster optimisation and completion')
        B_corrected = mnres.B


    # Output: Preparing output parameters as DataFrames
    if Map is None:
        Bout = op.clusters_B_as_dataframes(B_corrected, OGs, None)
    else:
        Bout, B_species = op.clusters_B_as_dataframes(B_corrected, OGs, MapNew)
    Xout = op.processed_X_as_dataframes(X_summarised_normalised, OGs, conditions)
    if input_is_one_dataset:
        Xout = Xout[0]

    # Output: Plot figures
    if showPlots:
        try:
            if np.shape(B_corrected)[1] > 0:
                graph.plotclusters(X_summarised_normalised, B_corrected, datafiles_noext, conditions, GDM=GDM,
                                   Cs='all', setPageToDefault=True, showPlots=showPlots, printToPDF=False)
        except:
            io.log('Error: could not generate clusters'' plots. Resuming the rest of steps ...')

    # Output: Prepare message to standard output and the summary then save the summary to a file and print the message
    summarymsg, endtime, timeconsumedtxt = \
        op.generateoutputsummaryparag(X, X_summarised_normalised, MapNew, GDMall, GDM,
                                      ures, mnres, B_corrected, starttime)
    io.log(summarymsg, addextrastick=False)

    io.deletetmpfile()

    return Bout, Xout, GDM
