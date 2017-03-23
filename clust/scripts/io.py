import os
import re
import numpy as np
import datastructures as ds
import glob
import output as op
import sys
import traceback
import math
import portalocker


def getFilesInDirectory(path, extension=None):
    for (dirpath, dirnames, filenames) in os.walk(path):
        if extension is None or extension == '':
            return [fn for fn in filenames]
        else:
            if len(extension) > 1 and extension[0] == '.' and extension[1] != '*':
                extension = extension[1:]
            return [fn for fn in filenames if re.match('(.*\.' + extension + '$)', fn) is not None]


def readDatasetsFromDirectory(path, delimiter='\t', skiprows=1, skipcolumns=1, returnSkipped=False):
    datafiles = np.sort(getFilesInDirectory(path)).tolist()

    datafileswithpath = [path + '/' + df for df in datafiles]

    datafilesread = readDataFromFiles(datafileswithpath, delimiter, float, skiprows, skipcolumns, returnSkipped)

    if returnSkipped:
        return datafilesread + (datafiles, )
    else:
        return datafilesread, datafiles


def readMap(mapfile, delimiter='\t'):
    if mapfile is None:
        return None
    return readDataFromFiles([mapfile], delimiter, dtype=str, skiprows=0, skipcolumns=0, returnSkipped=False)[0]


def readReplicates(replicatesfile, datafiles, replicates, delimiter='\t| |,|;'):
    if replicatesfile is None:
        if isinstance(replicates[0], list):
            return None, replicates
        elif isinstance(replicates[0], np.ndarray):
            return None, [r.tolist() for r in replicates]
        else:
            return None, replicates

    L = len(datafiles)
    replicatesIDs = [[-1 for c in replicates[l]] for l in range(L)]
    conditions = [None] * L

    with open(replicatesfile) as f:
        lineNumber = 0
        for line in f:
            lineNumber += 1
            line = line.partition('#')[0]
            line = line.rstrip()
            line = filter(None, re.split(delimiter, line))

            # Skip to next line if it is an empty line
            if len(line) < 1:
                continue

            if line[0] in datafiles:
                l = datafiles.index(line[0])  # (l)th dataset
            else:
                raise ValueError('Unrecognised data file name ({0}) in line {1} in {2}.'.
                                 format(line[0], lineNumber, replicatesfile))

            # Skip to next line if no condition ID is given
            if len(line) < 2:
                continue

            if conditions[l] is None:
                conditions[l] = [line[1]]
            elif line[1] not in conditions:
                conditions[l] += [line[1]]

            c = conditions[l].index(line[1])  # (c)th condition

            # Skip to next line if no replicates are given
            if len(line) < 3:
                continue

            for r in line[2:]:
                if r in replicates[l]:
                    if isinstance(replicates[l], list):
                        replicatesIDs[l][replicates[l].index(r)] = c
                    elif isinstance(replicates[l], np.ndarray):
                        replicatesIDs[l][replicates[l].tolist().index(r)] = c
                else:
                    raise ValueError('Unrecognised replicate name ({0}) in line {1} in {2}.'.
                                     format(r, lineNumber, replicatesfile))

    return (replicatesIDs, conditions)


def readNormalisation(normalisefile, datafiles, delimiter='\t| |,|;', defaultnormalisation=0):
    if normalisefile is None:
        return defaultnormalisation

    L = len(datafiles)
    normalise = [None] * L

    with open(normalisefile) as f:
        lineNumber = 0
        for line in f:
            lineNumber += 1
            line = line.partition('#')[0]
            line = line.rstrip()
            line = filter(None, re.split(delimiter, line))

            # Skip to next line if it is an empty line
            if len(line) < 1:
                continue

            if line[0] in datafiles:
                l = datafiles.index(line[0])  # (l)th dataset
            else:
                raise ValueError('Unrecognised data file name ({0}) in line {1} in {2}.'.
                                 format(line[0], lineNumber, normalisefile))

            # If no normalisation is set for the dataset, skip to the next line
            if len(line) < 2:
                continue

            # If the normalisation of this dataset has not been set, set it, otherwise
            if normalise[l] is None:
                normalise[l] = line[1:]
            else:
                normalise[l] = normalise[l] + line[1:]

    for l in range(L):
        if normalise[l] is None:
            normalise[l] = [0]
        else:
            normalise[l] = [int(n) for n in normalise[l]]

    return normalise


def readDataFromFiles(datafiles, delimiter='\t', dtype=float, skiprows=1, skipcolumns=1, returnSkipped=True, comm='#'):
    L = len(datafiles)
    X = [None] * L
    skippedRows = [None] * L
    skippedCols = [None] * L
    for l in range(L):
        with open(datafiles[l]) as f:
            ncols = len(f.readline().split(delimiter))
        X[l] = np.loadtxt(datafiles[l], delimiter=delimiter, dtype=dtype, skiprows=skiprows,
                          usecols=range(skipcolumns, ncols), ndmin=2, comments=comm)

        if skiprows > 0:
            skippedRows[l] = np.loadtxt(datafiles[l], delimiter=delimiter, dtype=str, skiprows=0,
                                        usecols=range(skipcolumns, ncols), comments=comm)[0:skiprows]
            if skiprows == 1:
                skippedRows[l] = skippedRows[l][0]
        else:
            skippedRows[l] = np.array([]).reshape([0, X[l].shape[1]])

        if skipcolumns > 0:
            skippedCols[l] = np.loadtxt(datafiles[l], delimiter=delimiter, dtype=str, skiprows=skiprows,
                                        usecols=range(skipcolumns), comments=comm)
        else:
            skippedCols[l] = np.array([]).reshape([0, X[l].shape[1]])

    if returnSkipped:
        return (ds.listofarrays2arrayofarrays(X), skippedRows, skippedCols)
    else:
        return ds.listofarrays2arrayofarrays(X)


def writedic(filepath, dic, header=None, delim='\t'):
    f = open(filepath, 'w+')

    # Write header
    if header is not None:
        f.write('{0}\n'.format(header))

    # Write the rest
    nokey = re.compile('^nokey[0-9]*$', flags=re.I)  # To match lines with no keys
    for k in dic.keys():
        if nokey.match(k) is None:
            f.write('{0}{1}{2}\n'.format(k, delim, dic[k]))
        else:
            f.write('{0}\n'.format(dic[k]))

    # Close file
    f.close()


def log(msg=None, addextrastick=True):
    if addextrastick:
        msg = op.msgformated(msg, withnewline=False)
    printOnConsole = True
    with open(glob.logfile, mode='a+') as f:
        if msg is not None:
            f.write(msg)
        f.write('\n')

    if printOnConsole:
        print(msg)


def logerror(exec_info):
    errstr = traceback.format_exception(exec_info[0], exec_info[1], exec_info[2])
    errstr = ''.join(errstr)
    log('Unexpected error:\n{0}\nContinuing execution anyway ...'.format(errstr))


def resetparallelprogress(parallel_total, log_every_percentage=10.0):
    with open(glob.tmpfile, mode='w+') as f:
        f.write('{0} {1} {2}'.format(parallel_total, log_every_percentage, 0.0))
        f.truncate()


def updateparallelprogress(added_value):
    Done = False
    while (not Done):
        try:
            Done = True
            with open(glob.tmpfile, mode='r+') as f:
                portalocker.lock(f, portalocker.LOCK_EX)
                data = f.read().split(" ")
                parallel_total = float(data[0])
                log_every_percentage = float(data[1])
                current_parallel_progress = float(data[2])

                last_log = math.floor(100 * current_parallel_progress
                                      / parallel_total / log_every_percentage) * log_every_percentage
                current_parallel_progress += added_value
                new_log = math.floor(100 * current_parallel_progress
                                     / parallel_total / log_every_percentage) * log_every_percentage

                #for i in np.arange(last_log+log_every_percentage, new_log + log_every_percentage, log_every_percentage):
                #    log('{0}%'.format(int(i)))
                if new_log > last_log:
                    log('{0}%'.format(int(new_log)))

                f.seek(0)
                f.write('{0} {1} {2}'.format(parallel_total, log_every_percentage, current_parallel_progress))
                f.truncate()
        except:
            Done = False


def getparallelprogress():
    Done = False
    while (not Done):
        try:
            Done = True
            with open(glob.tmpfile, mode='r+') as f:
                data = f.read().split(" ")
                parallel_total = float(data[0])
                log_every_percentage = float(data[1])
                current_parallel_progress = float(data[2])

                return (parallel_total, log_every_percentage, current_parallel_progress)
        except:
            Done = False


def deletetmpfile():
    os.remove(glob.tmpfile)

