import bicopam
import subprocess


datapath = '/home/basel/PycharmProjects/GE_data_grasses/Data'
mapfile = '/home/basel/PycharmProjects/GE_data_grasses/Map3000.txt'
repsfile = '/home/basel/PycharmProjects/GE_data_grasses/Replicates.txt'
normfile = '/home/basel/PycharmProjects/GE_data_grasses/Normalisation.txt'
outpath = None
#samplesIDs = [[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
#              [-1, -1, -1, -1, 0, 0, 1, 1, 2, 2],
#              [-1, -1, -1, -1, 0, 0, 1, 1, 2, 2]]
#normalise = [[101, 3, 4], [101, 3, 4], [101, 3, 4]]
Ks=[12, 15]
bicopam.bicopam(datapath, mapfile, repsfile, normfile, outpath, Ks, 5, 0.01, 3, 10)