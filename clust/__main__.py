import argparse
from argparse import RawTextHelpFormatter
import sys
import clustpipeline
from scripts.glob import version
import scripts.output as op


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # Parse arguments
    headertxt = op.topline()
    headertxt += op.msgformated('Clust\n'
                                'Optimised consensus clustering of multiple heterogeneous datasets\n'
                                'Version {0}\n'
                                '\n'
                                'By Basel Abu-Jamous\n'
                                'Department of Plant Sciences\n'
                                'The University of Oxford\n'
                                'basel.abujamous@plants.ox.ac.uk'.format(version), '^')
    headertxt += op.midline()
    headertxt += op.msgformated('Citation\n'
                                '~~~~~~~~', '^')
    citationtxt = 'When publishing work that uses Clust, please include these two citations:\n' \
                  '1. Basel Abu-Jamous and Steven Kelly (2018) Clust: automatic extraction of optimal co-expressed ' \
                  'gene clusters from gene expression data. bioRxiv 221309; doi: https://doi.org/10.1101/221309.\n' \
                  '2. Basel Abu-Jamous, Rui Fa, David J. Roberts, and Asoke K. Nandi (2013) Paradigm of tunable ' \
                  'clustering using binarisation of consensus partition matrices (Bi-CoPaM) for gene discovery, ' \
                  'PLOS ONE, 8(2): e56432'
    headertxt += op.msgformated(citationtxt, '<')
    headertxt += op.midline()
    headertxt += op.msgformated('Full description of usage can be found at:\n'
                                'https://github.com/BaselAbujamous/clust', '<')
    headertxt += op.bottomline()

    parser = argparse.ArgumentParser(description=headertxt, formatter_class=RawTextHelpFormatter)
    parser.add_argument('datapath', help='The directory that includes the data files.', default=None)
    parser.add_argument('-n', metavar='<file or int>', help='Normalisation file or list of codes (default: 1000)', default=['1000'], nargs='+')
    parser.add_argument('-r', metavar='<file>', help='Replicates structure file', default=None)
    parser.add_argument('-m', metavar='<file>', help='OrthoGroups (OGs) mapping file', default=None)
    parser.add_argument('-o', metavar='<directory>', help='Output directory', default=None)
    parser.add_argument('-t', metavar='<real number>', type=float,
                        help='Cluster tightness (default: 1.0).', default=1.0)
    parser.add_argument('-basemethods', metavar='<string>', nargs='+',
                        help='One or more base clustering methods (default: k-means)',
                        default=None)
    parser.add_argument('-K', metavar='<integer>', type=int, nargs='+',
                        help='K values, e.g. 2 4 6 10 ... (default: 4 to 20 (step=4))',
                        default=[n for n in range(4, 21, 4)])
    parser.add_argument('-s', metavar='<real number>', type=float,
                        help='Outlier standard deviations (default: 3.0)', default=3.0)
    parser.add_argument('-d', metavar='<integer>', type=int,
                        help='Min datasets in which a gene must exist (default: 1)', default=1)
    parser.add_argument('-fil-v', metavar='<real number>', dest='filv', type=float,
                        help='Filtering: gene expression threshold (default: -inf)', default=-float("inf"))
    parser.add_argument('-fil-c', metavar='<integer>', dest='filc', type=int,
                        help='Filtering: number of conditions (default: 0)',
                        default=0)
    parser.add_argument('-fil-d', metavar='<integer>', dest='fild', type=int,
                        help='Filtering: number of datasets (default: 0)', default=0)
    parser.add_argument('--fil-abs', dest='absval', action='store_true',
                        help='Filter using absolute values of expression')
    parser.add_argument('--fil-perc', dest='filperc', action='store_true',
                        help='-fil-v is a percentile of genes rather than raw value')
    parser.add_argument('--fil-flat', dest='filflat', action='store_true',
                        help='Filter out genes with flat expression profiles (default)')
    parser.add_argument('--no-fil-flat', dest='filflat', action='store_false',
                        help='Cancels the default --fil-flat option')
    parser.add_argument('-cs', metavar='<integer>', type=int, help='Smallest cluster size (default: 11)', default=11)
    parser.add_argument('-q3s', metavar='<real number>', type=float,
                        help='Q3''s defining outliers (default: 2.0)', default=2.0)
    parser.add_argument('--no-optimisation', dest='optimisation', action='store_false',
                        help='Skip cluster optimsation & completion')
    parser.add_argument('--deterministic', dest='deterministic', action='store_true',
                        help='Obsolete as all steps are already deterministic (v1.7.4+)')
    parser.add_argument('-np', metavar='<integer>', type=int, help='Number of parallel processes (default: 1)',
                        default=1)
    parser.set_defaults(optimisation=True, deterministic=False, absval=False, filperc=False, filflat=True)
    # parser.add_argument('-ec', type=int, help='Perform error correction, 1 or 0 (default: 1)', default=1)

    if len(args) == 0:
        parser.parse_args(['-h'])

    args = parser.parse_args(args)

    if args.filperc:
        filtype = 'perc'
    else:
        filtype = 'raw'

    if args.basemethods is not None:
        args.basemethods = [[m] for m in args.basemethods]

    # Call the clust function
    clustpipeline.clustpipeline(args.datapath, args.m, args.r, args.n, args.o, args.K, args.t,
                                args.s, args.d, args.filv, args.filc, args.fild, args.absval, filtype, args.filflat,
                                args.cs, args.np, args.optimisation, args.q3s, args.basemethods, args.deterministic)


if __name__ == "__main__":
    main()
