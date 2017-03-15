import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import glob

maxrows_per_page = 8
maxcols_per_page = 8
bands_per_page = 1
pagesize = (11.69, 8.27)
page_is_landscape = True
fontsize = 16
xticksrotation = 0


def set_maxrows_per_page(val):
    global maxrows_per_page
    maxrows_per_page = val


def set_maxcols_per_page(val):
    global maxcols_per_page
    maxcols_per_page = val


def set_bands_per_page(val):
    global bands_per_page
    bands_per_page = val


def set_pagesize(val):
    global pagesize
    pagesize = val


def set_page_is_landscape(val, setRowsColsToDefault=False):
    global page_is_landscape
    page_is_landscape = val
    if setRowsColsToDefault:
        if page_is_landscape:
            set_maxrows_per_page(8)
            set_maxcols_per_page(8)
        else:
            set_maxrows_per_page(12)
            set_maxcols_per_page(5)
    if page_is_landscape:
        set_pagesize((11.69, 8.27))
    else:
        set_pagesize((8.27, 11.69))


def set_fontsize(val):
    global fontsize
    fontsize = val


def set_xticksrotation(val):
    global xticksrotation
    xticksrotation = val


def set_best_fit_page_parameters(L, K):
    if (L / float(K)) >= (12.0 / 5.0):
        set_page_is_landscape(False, setRowsColsToDefault=True)
    else:
        set_page_is_landscape(True, setRowsColsToDefault=True)

    if maxrows_per_page == L:
        set_bands_per_page(1)
    elif maxrows_per_page > L:
        set_bands_per_page((maxrows_per_page + 1)/ (L + 1))  # 1, 2, 3, 4, etc (int)
    else:
        set_bands_per_page(1.0 / math.ceil(float(L) / maxrows_per_page))  # 0.5, 0.33, 0.25, 0.2, etc (float)

    number_of_bands = int(math.ceil(float(K) / maxcols_per_page))
    set_maxcols_per_page(int(math.ceil(float(K) / number_of_bands)))
    if bands_per_page == 1:
        set_maxrows_per_page(L)
    elif bands_per_page > 1:
        if K <= maxcols_per_page * bands_per_page:
            set_maxrows_per_page((L + 1) * number_of_bands - 1)
            set_bands_per_page(number_of_bands)
        else:
            set_maxrows_per_page((L + 1) * bands_per_page - 1)
    else:
        set_maxrows_per_page(int(math.ceil(L * bands_per_page)))
        #set_maxrows_per_page(int(math.ceil((L + 1 / bands_per_page) * bands_per_page)))

    Nplots = maxrows_per_page * maxcols_per_page
    set_fontsize(int(0.6 * (110 - Nplots) / 5.0))


def position_of_subplot(L, K, l, k):
    band = k / maxcols_per_page  # Number of band relative to the beginning of the plots (0, 1, 2, 3, ...)
    page = int(band / bands_per_page)  # Page 0, 1, 2, 3, ...
    if bands_per_page < 1:
        page += l / maxrows_per_page
    band_in_page = band - int(page * bands_per_page)  # Band within page 0, 1, 2, ... (bands_per_page - 1)

    col = k % maxcols_per_page
    if bands_per_page == 1:
        row = l
    elif bands_per_page > 1:
        row = band_in_page * (L + 1) + l
    else:
        row = band_in_page * L + (l % maxrows_per_page)
    pos = row * maxcols_per_page + col + 1  # position of the subplot in the plot

    return (page, pos, row, col)


def plotclusters(X, B, filename, DatasetsNames, conditions, GDM=None, Cs='all', setPageToDefault=True):
    plt.ioff()  # Turn iteractive mode off so the figures do not show up without calling .show()
    if isinstance(Cs, basestring) and Cs == 'all':
        K = B.shape[1]  # Number of clusters to be plotted
        Cs = [c for c in range(K)]  # Clusters to be plotted
    else:
        K = len(Cs)  # Number of clusters to be plotted

    L = len(X)  # Number of datasets

    if setPageToDefault:
        set_best_fit_page_parameters(L, K)  # Landscape or portrait, cols and rows, fonts, etc.
    matplotlib.rcParams.update({'font.size': fontsize})  # Set font size
    matplotlib.rcParams.update({'xtick.labelsize': 0.7 * fontsize})  # Set font size for the xticks
    Np = position_of_subplot(L, K, L-1, K-1)[0] + 1  # Number of pages

    # Prepare plots (figure(0) to figure(Np-1)
    for k in range(K):
        for l in range(L):
            (page, pos, row, col) = position_of_subplot(L, K, l, Cs[k])
            plt.figure(page, figsize=pagesize, frameon=False)
            plt.subplot(maxrows_per_page, maxcols_per_page, pos)
            # Get the subset of the dataset relevant to this cluster and this dataset
            if GDM is None:
                localX = X[l][B[:, Cs[k]], :]
            else:
                localX = X[l][B[GDM[:, l], Cs[k]], :]
            plt.plot(np.arange(localX.shape[1]), np.transpose(localX), 'k-')
            plt.xticks(np.arange(localX.shape[1]), conditions[l], rotation=xticksrotation)
            # Suppress ticks
            frame = plt.gca()
            frame.axes.get_yaxis().set_ticks([])

            '''
            if l < L - 1 and l < maxrows_per_page - 1:
                frame.axes.get_xaxis().set_ticks([])
            else:
                frame.axes.get_xaxis().set_ticks([i for i in range(localX.shape[1])])
            '''

            # Add title
            if l == 0 or row == 0:
                plt.title('C{0}\n({1} {2})'.format(Cs[k], np.sum(B[:, Cs[k]]), glob.object_label_lower))
            # Add datasets names (ylabels)
            if col == 0:
                plt.ylabel(DatasetsNames[l])

    # Save plots
    with PdfPages(filename) as pdf:
        for p in range(Np):
            #plt.figure(p)
            #plt.clf()
            pdf.savefig(figure=p)
            pdf.attach_note('Page {0}'.format(p + 1))
        info = pdf.infodict()
        info['Author'] = 'Clust python package'
    plt.close('all')

