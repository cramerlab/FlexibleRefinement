import numpy as np
import matplotlib.pyplot as plt
import os

arr = np.loadtxt(os.path.join("bin", "Debug", "Refinement", "current", "1DCTFs.txt"), delimiter="\t")

plt.figure()
plt.title("1DCTFs")
plt.plot(arr[0,:])


arr = np.loadtxt(os.path.join("bin", "Debug", "Refinement", "current", "GaussTableConvolved.txt"), delimiter="\t")
plt.figure()
plt.title("GaussTableConvolved")
plt.plot(arr[0,:])



arr = np.loadtxt(os.path.join("bin", "Debug", "Refinement", "current", "GaussTable.txt"), delimiter="\t")
plt.figure()
plt.title("GaussTable")
plt.plot(arr[0,:])

arr = np.loadtxt(os.path.join("bin", "Debug", "Refinement", "current", "GaussTableFT.txt"), delimiter="\t")
plt.figure()
plt.title("GaussTableFT")
plt.plot(arr[0,0:500])
plt.show()



