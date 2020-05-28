from glob import glob
import matplotlib.pyplot as plt
import re
import sys
import os

def findIntersect(a, x1,y1,x2,y2):

    return (a-(x2*y1-x1*y2)/(x2-x1))*(x2-x1)/(y2-y1)

def getFSCCutoffs(fscFile):
    pixels = []
    values = []
    error = False;
    found1 = False
    found5 = False
    with open(fscFile, "r") as infile:

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
            if not found1 and values[-1] <= 0.143:
                cutOff1 = findIntersect(0.143, pixels[-2], values[-2], pixels[-1], values[-1])
                found1 = True

    if error:
        sys.stderr.write("Error processing file {}. Possibly contains nan values\n".format(f))
        return float("nan"), float("nan")
    boxsize = pixels[-1] * 2
    pixels = [p * 1 / (boxsize * voxelSize) for p in pixels]

    if found5:
        cutOff5 = cutOff5 * 1 / (boxsize * voxelSize)
    else:
        cutOff5 = pixels[-1]
    if found1:
        cutOff1 = cutOff1 * 1 / (boxsize * voxelSize)
    else:
        cutOff1 = pixels[-1]

    return cutOff1, cutOff5

voxelSize = 1.5
k = 75
inPrefix = r"D:\EMD\9233\emd_9233_Scaled_1.5_{}k_bs64_it*_oversampled*_masked.fsc".format(k)


initialRep = glob(r"D:\EMD\9233\emd_9233_Scaled_1.5_{}k_bs64_initialRep_oversampled*_masked.fsc".format(k))[0]
cutOff1, cutOff5 = getFSCCutoffs(initialRep)
its = [0]
res = [1/cutOff5]
for f in glob(inPrefix):
    it = int(re.search(r"it(\d+)", f).group(1))
    cutOff1, cutOff5 = getFSCCutoffs(f)
    its.append(it)
    res.append(1/cutOff5)
plt.figure()
plt.plot(its[1:], res[1:], "*")
plt.ylim([0, its[1] + 1])
plt.xlabel("Iteration")
plt.ylabel("Resolution in Å")
plt.savefig(r"D:\EMD\9233\emd_9233_Scaled_1.5_{}k_bs64_perItRes.png".format(k))
