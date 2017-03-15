import clust

datapath = '/home/basel/PycharmProjects/bicopam/ExampleData/2_Next_level/Data'
mapfile = '/home/basel/PycharmProjects/bicopam/ExampleData/2_Next_level/MapIDs.tsv'
repsfile = '/home/basel/PycharmProjects/bicopam/ExampleData/2_Next_level/Replicates.txt'
normfile = '/home/basel/PycharmProjects/bicopam/ExampleData/2_Next_level/Normalisation.txt'
outpath = None
#samplesIDs = [[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
#              [-1, -1, -1, -1, 0, 0, 1, 1, 2, 2],
#              [-1, -1, -1, -1, 0, 0, 1, 1, 2, 2]]
#normalise = [[101, 3, 4], [101, 3, 4], [101, 3, 4]]
#Ks=[12, 15]
clust.clust(datapath, mapfile, repsfile, normfile, tightnessweight=1)