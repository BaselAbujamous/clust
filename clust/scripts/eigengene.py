import numpy as np
import pandas as pd
import scipy.stats as spst


def cluster_eigengene(localX):
    svd = np.linalg.svd(localX)
    eigengene = svd[2][0]*(localX.shape[1]-1)**0.5
    # Align sign with the average expression profile:
    avr_expression = np.average(localX, axis=0)
    if spst.pearsonr(eigengene, avr_expression)[0] < 0:
        eigengene = -eigengene
    return eigengene


def eigengenes_dataframe(X_summarised_normalised, B_corrected, conditions):
    K = B_corrected.shape[1]
    Cs = ['C'+ str(c) for c in range(K)]
    eigengene_list = []
    for k in range(K):
        kcluster = B_corrected[:,k]
        localX = X_summarised_normalised[0][kcluster]
        keigengene = cluster_eigengene(localX)
        eigengene_list.append(keigengene)
    return pd.DataFrame(data=np.vstack(eigengene_list), index=Cs, columns=conditions[0])
