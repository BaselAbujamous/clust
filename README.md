# Bi-CoPaM
The Bi-CoPaM identifies clusters (groups) of objects which are well-correlated with each other across a number of given datasets with minimal need for manual intervention.
![Clusters](https://github.com/BaselAbujamous/bicopam/blob/master/Clusters.png)
*Figure 1: The Bi-CoPaM generates clusters (C0, C1, C2, ...) out of an input of 9,462 objects based on their profiles in three datasets (X0, X1, and X2). The left-hand panel shows the profiles of all 9,462 objects in each one of the three datasets, while the right-hand panel shows the profiles of the objects within each one of the clusters. The objects included in any given cluster are well-correlated with each other in each one of the three datasets. Note that the number of conditions or time points are different amongst the datasets.*

**Features!**

1. No need to filter your data before submission.
2. No need to preset the number of clusters; the algorithm finds it automatically.
3. The algorithm automatically filters out any objects that do not fit into any cluster.
4. You can control the tightness of the clusters simply by varying a parameter, which has a default value if you wish not to set it!
5. You can include heterogeneous datasets (e.g. gene expression datasets from different technologies, different species, different numbers of conditions, etc). Have a look at the **Simple usage** section.
5. The package calculates key statistics and provides them in the output.
6. A table of clusters' members is provided in an output TSV file.
7. A figure showing the profiles of the generated clusters is provided as an output PDF file.

## Automatic Bi-CoPaM analysis pipeline
![Bi-CoPaM workflow](https://github.com/BaselAbujamous/bicopam/blob/master/Workflow_PyPkg.png)
*Figure 2: Automatic Bi-CoPaM analysis pipeline*

## Simplest usage
- `bicopam data_path`

#### Data files

## Simple usage
- `bicopam data_path -m map_file -n normalisation_file -r replicates_file -t tightness`

#### Map file

#### Normalisation file

#### Replicates file

## Advanced usage

