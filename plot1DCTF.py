import numpy as np
import matplotlib.pyplot as plt
import os
'''
arr = np.loadtxt(os.path.join("bin", "Debug", "Refinement", "current", "GaussTable.txt"), delimiter="\t")
plt.figure()
plt.title("GaussTable")
plt.plot(arr[:])


arr = np.loadtxt(os.path.join("bin", "Debug", "Refinement", "current", "1DCTFs.txt"), delimiter="\t")

plt.figure()
plt.title("1DCTFs")
plt.plot(arr[:])


arr = np.loadtxt(os.path.join("bin", "Debug", "Refinement", "current", "GaussTableConvolved.txt"), delimiter="\t")
plt.figure()
plt.title("GaussTableConvolved")
plt.plot(arr[::10])


arr = np.loadtxt(os.path.join("bin", "Debug", "Refinement", "current", "testImageConvolvedSelf.txt"), delimiter="\t")
plt.figure()
plt.title("testImageConvolvedSelf")
plt.plot(arr[:])




arr = np.loadtxt(os.path.join("bin", "Debug", "Refinement", "current", "testImageConvolved.txt"), delimiter="\t")
plt.figure()
plt.title("testImageConvolved")
plt.plot(arr[:])





arr = np.loadtxt(os.path.join("bin", "Debug", "Refinement", "current", "testImageFT.txt"), delimiter="\t")
plt.figure()
plt.title("testImageFT")
plt.plot(arr[:])


arr = np.loadtxt(os.path.join("bin", "Debug", "Refinement", "current", "testImageSelfConstructed.txt"), delimiter="\t")
plt.figure()
plt.title("testImageSelfConstructed")
plt.plot(arr[:])


arr = np.loadtxt(os.path.join("bin", "Debug", "Refinement", "current", "GaussTableFT.txt"), delimiter="\t")
plt.figure()
plt.title("GaussTableFT")
plt.plot(arr[::1])

'''
arr = np.loadtxt(os.path.join("bin", "Debug", "Refinement", "20200127_9204feb", "testImage2.txt"), delimiter="\t")
plt.figure()
plt.title("testImage2")
plt.plot(arr[:])


arr = np.loadtxt(os.path.join("bin", "Debug", "Refinement", "20200127_9204feb", "testImage2SelfConvolved.txt"), delimiter="\t")
plt.figure()
plt.title("testImage2SelfConvolved")
plt.plot(arr[:])
arr = np.loadtxt(os.path.join("bin", "Debug", "Refinement", "20200127_9204feb", "testImage2Convolved.txt"), delimiter="\t")
plt.figure()
plt.title("testImage2Convolved")
plt.plot(arr[:])




plt.show()



