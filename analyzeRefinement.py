import numpy as np
import matplotlib.pyplot as plt
import os
import sys


itErrs = []
with open(r"D:\Software\FlexibleRefinement\bin\Debug\Refinement\itErr.txt", "r") as errFile:
    it = -1
    for line in errFile:
        line = line.strip()
        if not line:
            continue
        if line.startswith("it"):
            it = it + 1
            itErrs.append([])
            continue
        err = float(line.replace(",", '.'))**2
        itErrs[it].append(err)


plt.boxplot(itErrs)
plt.show()
print("done")