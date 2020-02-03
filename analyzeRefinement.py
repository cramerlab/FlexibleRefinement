import numpy as np
import matplotlib.pyplot as plt
import os
import sys


itErrsCTF = []
with open(r"D:\Software\FlexibleRefinement\bin\Debug\Refinement\current\itErrsCTF.txt", "r") as errFile:
    it = -1
    for line in errFile:
        line = line.strip()
        if not line:
            continue
        if line.startswith("it"):
            it = it + 1
            itErrsCTF.append([])
            continue
        err = np.sqrt(float(line.replace(",", '.'))**2)
        itErrsCTF[it].append(err)


itDiffsCTF = []
with open(r"D:\Software\FlexibleRefinement\bin\Debug\Refinement\current\itDiffsCTF.txt", "r") as errFile:
    it = -1
    for line in errFile:
        line = line.strip()
        if not line:
            continue
        if line.startswith("it"):
            it = it + 1
            itDiffsCTF.append([])
            continue
        err = np.sqrt(float(line.replace(",", '.'))**2)
        itDiffsCTF[it].append(err)

itErrs = []
with open(r"D:\Software\FlexibleRefinement\bin\Debug\Refinement\current\itErrsCTF.txt", "r") as errFile:
    it = -1
    for line in errFile:
        line = line.strip()
        if not line:
            continue
        if line.startswith("it"):
            it = it + 1
            itErrs.append([])
            continue
        err = np.sqrt(float(line.replace(",", '.'))**2)
        itErrs[it].append(err)


itDiffs = []
with open(r"D:\Software\FlexibleRefinement\bin\Debug\Refinement\current\itDiffsCTF.txt", "r") as errFile:
    it = -1
    for line in errFile:
        line = line.strip()
        if not line:
            continue
        if line.startswith("it"):
            it = it + 1
            itDiffs.append([])
            continue
        err = np.sqrt(float(line.replace(",", '.'))**2)
        itDiffs[it].append(err)

plt.figure()
plt.plot(itErrsCTF, "*")
plt.yscale('log')
plt.title("itErrsCTF")

plt.figure()
plt.plot(itErrs, "*")
plt.yscale('log')
plt.title("itErrs")

plt.figure()
plt.figure()
plt.boxplot(itDiffs)
plt.yscale('log')
plt.title("itDiffs")

plt.figure()
plt.boxplot(itDiffsCTF)
plt.yscale('log')
plt.show()
print("done")