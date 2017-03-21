import argparse
from argparse import RawTextHelpFormatter
import sys
import clust.clust

args = sys.argv[1:]

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
parser.add_argument('datapath', help='The path of the data files.', default=None)
parser.add_argument('-r', help='Replicates file path', default=None)
parser.add_argument('-n', help='Normalisation file path', default=None)
parser.add_argument('-m', help='Object-groups (OGs) mapping file path', default=None)
parser.add_argument('-o', help='Output directory', default=None)
parser.add_argument('-K', type=int, nargs='+', help='K values (default: all values from 2 to 20 inclusively)',
                    default=[n for n in range(2, 21)])
parser.add_argument('-t', type=float, help='Cluster tightness versus cluster size weight: '
                                           'a real positive number, where 1.0 means equal weights '
                                           '(default: 1.0).', default=1.0)
parser.add_argument('-s', type=float, help='Number of standard deviations that define an outlier '
                                           '(default: 3.0)', default=3.0)
parser.add_argument('-d', type=int, help='Minimum number of datasets that an object has to be included in for '
                                         'it to be considered in Clust analysis. If an object is included '
                                         'only in fewer datasets than this, it will be excluded from the analysis '
                                         '(default: 1)', default=1)
parser.add_argument('-fil-v', dest='filv', type=float,
                    help='Data value (e.g. gene expression) threshold. Any value lower than this will be set to '
                         '0.0. If an object never exceeds this value at least in -fil-c conditions in at least '
                         '-fil-d datasets, it is excluded from the analysis (default: -inf)', default=-float("inf"))
parser.add_argument('-fil-c', dest='filc', type=int,
                    help='Minimum number of conditions in a dataset in which an object should exceed the data '
                         'value -fil-v at least in -fil-d datasets to be included in the analysis (default: 0)',
                    default=0)
parser.add_argument('-fil-d', dest='fild', type=int,
                    help='Minimum number of datasets in which an object should exceed the data value -fil-v at '
                         'least in -fil-c conditions to be included in the analysis (default: 0)', default=0)

parser.add_argument('-cs', type=int, help='Smallest cluster size (default: 11)', default=11)
# parser.add_argument('-ec', type=int, help='Perform error correction, 1 or 0 (default: 1)', default=1)

if len(args) == 0:
    parser.parse_args(['-h'])

args = parser.parse_args(args)

# Call the clust function
clust.clust.clust(args.datapath, args.m, args.r, args.n, args.o, args.K, args.t,
                  args.s, args.d, args.filv, args.filc, args.fild, args.cs)

