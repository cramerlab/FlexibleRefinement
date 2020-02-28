import numpy as np
import matplotlib.pyplot as plt
import os
import sys


input = "lowHighMix_small"

itErrsCTF = []
with open(r"D:\Software\FlexibleRefinement\bin\Debug\Refinement\current\{}\itErrsCTF.txt".format(input), "r") as errFile:
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
with open(r"D:\Software\FlexibleRefinement\bin\Debug\Refinement\current\{}\itDiffsCTF.txt".format(input), "r") as errFile:
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

itErrsMoved = []
with open(r"D:\Software\FlexibleRefinement\bin\Debug\Refinement\current\{}\itErrsMoved.txt".format(input), "r") as errFile:
    it = -1
    for line in errFile:
        line = line.strip()
        if not line:
            continue
        if line.startswith("it"):
            it = it + 1
            itErrsMoved.append([])
            continue
        err = np.sqrt(float(line.replace(",", '.'))**2)
        itErrsMoved[it].append(err)

itCorrsCTF = []
with open(r"D:\Software\FlexibleRefinement\bin\Debug\Refinement\current\{}\itCorrsCTF.txt".format(input), "r") as errFile:
    it = -1
    for line in errFile:
        line = line.strip()
        if not line:
            continue
        if line.startswith("it"):
            it = it + 1
            itCorrsCTF.append([])
            continue
        for val in (line.replace(",", ".").split(" ")):
            err = np.sqrt(float(val)**2)
            itCorrsCTF[it].append(err)


itDiffsMoved = []
with open(r"D:\Software\FlexibleRefinement\bin\Debug\Refinement\current\{}\itDiffsMoved.txt".format(input), "r") as errFile:
    it = -1
    for line in errFile:
        line = line.strip()
        if not line:
            continue
        if line.startswith("it"):
            it = it + 1
            itDiffsMoved.append([])
            continue
        err = np.sqrt(float(line.replace(",", '.'))**2)
        itDiffsMoved[it].append(err)


itCorrsMoved = []
with open(r"D:\Software\FlexibleRefinement\bin\Debug\Refinement\current\{}\itCorrsMoved.txt".format(input), "r") as errFile:
    it = -1
    for line in errFile:
        line = line.strip()
        if not line:
            continue
        if line.startswith("it"):
            it = it + 1
            itCorrsMoved.append([])
            continue
        for val in (line.replace(",", ".").split(" ")):
            err = np.sqrt(float(val)**2)
            itCorrsMoved[it].append(err)

plt.figure()
plt.boxplot(itErrsCTF, "*")
plt.yscale('log')
plt.title("itErrsCTF")

plt.figure()
plt.boxplot(itDiffsCTF)
plt.yscale('log')
plt.title("itDiffsCTF")

plt.figure()
plt.boxplot(itCorrsCTF)
plt.yscale('log')
plt.title("itCorrsCTF")

plt.figure()
plt.boxplot(itErrsMoved, "*")
plt.yscale('log')
plt.title("itErrsMoved")

plt.figure()
plt.boxplot(itDiffsMoved)
plt.yscale('log')
plt.title("itDiffsMoved")

plt.figure()
plt.boxplot(itCorrsMoved)
plt.yscale('log')
plt.title("itCorrsMoved")


plt.show()
print("done")