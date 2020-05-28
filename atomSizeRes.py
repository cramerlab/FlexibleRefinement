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
inPrefix = r"D:\EMD\9233\emd_9233_Scaled_1.5_*k"

print("{:^7}\t{:^6}\t{:^9}".format("N", "σ [Å]", "FSC 0.5"))

sizes = []
res = []
for f in glob(inPrefix + ".pdb"):
    match = re.search(r"_(\d+)k", f)
    N = 1000 * int(match.group(1))
    size = 0.0
    with open(f, "r") as initialPDB:
        for line in initialPDB:
            if line.startswith("REMARK fixedGaussian"):
                sigma = float(line.split()[2])
                size = 1.5*sigma
                break
    fList = glob(f.replace(".pdb", "") + "*it*oversampled*_masked.fsc")
    largestItFile = ""
    largestIt = -1
    for f2 in fList:
        match = re.search(r"it(\d+)", f2)
        if match:
            it = int(match.group(1))
            if it > largestIt:
                largestItFile = f2
                largestIt = it
    if not largestItFile:
        continue
    cutOff1, cutOff5 = getFSCCutoffs(largestItFile)
    sizes.append(size)
    res.append(1/cutOff5)
    print("{:7.1e}\t{:^6.3f}\t{:^9.2f}".format(N, size, 1/cutOff5))
plt.plot(sizes, res, "*")
#plt.ylim([0,max(res)+0.2])
plt.xlabel("Radius of pseudo atoms in Å")
plt.ylabel("Resolution in Å")
plt.savefig(os.path.join(inPrefix.replace("*k", "atomSizeRes.png")))