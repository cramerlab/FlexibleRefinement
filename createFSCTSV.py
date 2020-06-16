import os
import sys
import matplotlib.pyplot as plt
from glob import glob
import re
#infileDir = r"D:\EMD\9233\emd_9233_Scaled_1.5_75k_bs64_it3_moving"
infileDir = r"D:\EMD\9233\AtomNumberSweep"
voxelSize = 1.5


def findIntersect(a, x1,y1,x2,y2):

    return (a-(x2*y1-x1*y2)/(x2-x1))*(x2-x1)/(y2-y1)


with open(r"D:\EMD\9233\AtomNumberSweep\fscs.tsv", "w") as outfile:
    outfile.write("{}\t{}\t{}\t{}\n".format("ATOMCOUNT", "OVERSAMPLING", "ROUND", "RESOLUTION"))
    for f in glob(os.path.join(infileDir, r"*.fsc"), recursive=True):
        N = int(re.search("N(\d+)", f).group(1))*1000
        OS = float(re.search("OS(\d\.\d)", f).group(1))
        m = re.search("round(\d)", f)
        round =1
        if m:
            round = int(m.group(1))
        pixels = []
        values = []
        error = False;
        found1 = False
        found5 = False
        cutOff5 = 0.0
        voxelSize = 1.5#float(re.search("Scaled_(\d\.\d)", f).group(1))
        with open(f, "r") as infile:

            for line in infile:
                if not line.strip():
                    continue
                splitted = line.split()
                pixels.append(int(splitted[0]))
                try:
                    values.append(float(splitted[1]))
                except ValueError:
                    error = True
                    break
                if not found5 and values[-1] <= 0.5:
                    cutOff5 = findIntersect(0.5, pixels[-2], values[-2], pixels[-1], values[-1])
                    found5 = True
                #if not found1 and values[-1] <= 0.143:
                #    cutOff1 = findIntersect(0.143, pixels[-2], values[-2], pixels[-1], values[-1])
                #    found1 = True

        if error:
            sys.stderr.write("Error processing file {}. Possibly contains nan values\n".format(f))
            continue
        boxsize = pixels[-1] * 2
        pixels = [p * 1 / (boxsize * voxelSize) for p in pixels]

        if found5:
            cutOff5 = cutOff5 * 1 / (boxsize * voxelSize)
        else:
            cutOff5 = pixels[-1]

        outfile.write("{}\t{}\t{}\t{}\n".format(N, OS, round, 1 / cutOff5))


