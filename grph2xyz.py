
from glob import glob
import numpy as np
import os
import re

def grph2xyz(graphFileName, xyzFileName):
    with open(graphFileName, "r") as ifile:
        lines = ifile.readlines()
        lines = lines[6:]
        centers = np.array([[float(f) for f in line.split()[:3]] for line in lines])

    with open(xyzFileName, "w") as ofile:
        ofile.write("{}\n".format(centers.shape[0]))
        ofile.write("{}\n".format(os.path.basename(graphFileName)))
        for i in range(centers.shape[0]):
            ofile.write("C {} {} {}\n".format(centers[i,0], centers[i,1],centers[i,2]))

for f in glob(r"D:\Software\FlexibleRefinement\bin\Debug\lennardJones\GridSearchParams_No_NeighborUpdate_c10000_it100\**\*.graph",recursive=True):
    if re.search(r"it\d+\.graph", f):
        continue
    grph2xyz(f, f.replace(".graph", ".xyz"))
#for f in glob(r"D:\Software\FlexibleRefinement\bin\Debug\lennardJones\differentRotExp\*.graph",recursive=True):
#    grph2xyz(f, f.replace(".graph", ".xyz"))