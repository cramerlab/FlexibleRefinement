import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os
import pandas as pd

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "Debug")

def plotDistanceHist(fullPath):
    distances = []
    with open(fullPath, "r") as dFile:
        for line in dFile:
            if not line.strip():
                continue
            distances.append(float(line))
    plt.hist(distances)
    plt.show()


def plotDisplacement(fullPath):
    displacements = []
    with open(fullPath, "r") as dFile:
        for line in dFile:
            if not line.strip():
                continue
            displacements.append([float(l) for l in line.split()])

    plt.hist(np.linalg.norm(np.array(displacements), axis=1))
    plt.show()


# plotDistanceHist(os.path.join(path, "Stick_to_Arc", "Stick_to_Arc_distanceList_initial.txt"))

def getCenters(fileName):
    with open(fileName, "r") as ifile:
        lines = ifile.readlines()
        lines = lines[6:]
        centers = np.array([[float(f) for f in line.split()[:3]] for line in lines])
    return centers




#for i in range(10):
#    plotDisplacement(os.path.join(path, "Stick_to_Arc", "Stick_to_Arc_displacementList_{}.txt".format(i)))
    # plotDistanceHist(os.path.join(path, "Stick_to_Arc", "Stick_to_Arc_distanceList_{}.txt".format(i)))

def plotDisplacements(fileFrom, fileTo):
    gtStartCenters = getCenters(fileFrom)
    gtEndCenters = getCenters(fileTo)
    X = gtStartCenters[::5,0]
    Y = gtStartCenters[::5, 1]
    Z = gtStartCenters[::5, 2]
    directions = gtEndCenters - gtStartCenters
    U = directions[::5,0]
    V = directions[::5, 1]
    W = directions[::5, 2]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.quiver(X, Y, Z, U, V, W)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_zlim([0, 100])
    plt.show()

def rotationExp():
    c = 20
    gtStartCenters = getCenters(r"D:\Software\FlexibleRefinement\bin\Debug\RotateStick\Stick_Initial.graph")
    gtEndCenters = getCenters(r"D:\Software\FlexibleRefinement\bin\Debug\RotateStick\Rotate_PI_{}_gt.graph".format(c))
    gtDisplacements = gtEndCenters - gtStartCenters
    results = []
    for corrScale in range(1, 21):
        for distScale in range(1, 21):
            for norm in ["True", "False"]:
                if not os.path.isfile(
                        r"D:\Software\FlexibleRefinement\bin\Debug\RotateStick\Rotate_PI_{}_final_{}_{}_{}.graph".format(
                                c, corrScale, distScale, norm)):
                    print("Rotate_PI_20_final_{}_{}_{}.graph not done yet".format(corrScale, distScale, norm))
                    results.append(
                        ["Rotate_PI_{}_final_{}_{}_{}".format(c, corrScale, distScale, norm), 0, 0,
                         0, 0])
                    continue
                endCenters = getCenters(
                    r"D:\Software\FlexibleRefinement\bin\Debug\RotateStick\Rotate_PI_{}_final_{}_{}_{}.graph".format(c,
                                                                                                                     corrScale,
                                                                                                                     distScale,
                                                                                                                     norm))
                displacements = endCenters - gtStartCenters
                offsets = endCenters - gtEndCenters
                offNorm = np.linalg.norm(offsets, axis=1)
                offMean = np.mean(offNorm)
                offStd = np.std(offNorm)
                dispDiff = displacements - gtDisplacements
                dispDiffNorm = np.linalg.norm(dispDiff, axis=1)
                dispDiffMean = np.mean(dispDiffNorm)
                dispDiffStd = np.std(dispDiffNorm)
                results.append(
                    ["Rotate_PI_{}_final_{}_{}_{}".format(c, corrScale, distScale, norm), offMean, offStd, dispDiffMean,
                     dispDiffStd])
                # plt.hist(diff)
    df = pd.DataFrame(results, columns=["Filename", "Mean Pos Offset", "Std Pos Offset", "Diff of distance", "std"])
    df.to_csv('list.csv', index=False)

def dosnwampledRotationExp():
    c = 10

    step = "StepThree"
    if step == "StepOne":
        factor = 4
    elif step == "StepTwo":
        factor = 2
    elif step == "StepThree":
        factor = 1
    gtStartCenters = getCenters(r"D:\Software\FlexibleRefinement\bin\Debug\downsampling\{}_StartGraph.graph".format(factor))
    gtEndCenters = getCenters(r"D:\Software\FlexibleRefinement\bin\Debug\downsampling\{}_TarGraph.graph".format(factor))
    gtDisplacements = gtEndCenters - gtStartCenters
    results = []
    for corrScale in range(1, 21):
        for distScale in range(1, 21):
            for norm in ["True", "False"]:
                if not os.path.isfile(
                        r"D:\Software\FlexibleRefinement\bin\Debug\downsampling\{}\{}_Rotate_PI_{}__{}_{}_{}_final.graph".format(
                                step, factor, c, corrScale, distScale, norm)):
                    print("{}_Rotate_PI_{}__{}_{}_{}_final.graph not done yet".format(factor, c, corrScale, distScale, norm))
                    results.append(
                        ["{}_Rotate_PI_{}_{}_{}_{}".format(factor, c, corrScale, distScale, norm), 0, 0,
                         0, 0])
                    continue
                endCenters = getCenters(
                    r"D:\Software\FlexibleRefinement\bin\Debug\downsampling\{}\{}_Rotate_PI_{}__{}_{}_{}_final.graph".format(
                        step,
                        factor,
                        c,
                        corrScale,
                        distScale,
                        norm))

                displacements = endCenters - gtStartCenters
                offsets = endCenters - gtEndCenters
                offNorm = np.linalg.norm(offsets, axis=1)
                offMean = np.mean(offNorm)
                offStd = np.std(offNorm)
                dispDiff = displacements - gtDisplacements
                dispDiffNorm = np.linalg.norm(dispDiff, axis=1)
                dispDiffMean = np.mean(dispDiffNorm)
                dispDiffStd = np.std(dispDiffNorm)
                results.append(
                    ["{}_Rotate_PI_{}_{}_{}_{}".format(factor, c, corrScale, distScale, norm), offMean, offStd, dispDiffMean,
                     dispDiffStd])
                # plt.hist(diff)
    df = pd.DataFrame(results, columns=["Filename", "Mean Pos Offset", "Std Pos Offset", "Diff of distance", "std"])
    df.to_csv('{}.csv'.format(step), index=False)
    '''
    gtStartCenters = getCenters(r"D:\Software\FlexibleRefinement\bin\Debug\downsampling\StartGraph_downSampled_2.graph")
    gtEndCenters = getCenters(r"D:\Software\FlexibleRefinement\bin\Debug\downsampling\TargetGraph_downSampled_2.graph")
    gtDisplacements = gtEndCenters - gtStartCenters
    results = []
    for corrScale in range(1, 21):
        for distScale in range(1, 21):
            for norm in ["True", "False"]:
                if not os.path.isfile(
                        r"D:\Software\FlexibleRefinement\bin\Debug\downsampling\StepTwo\downSampled_2_Rotate_PI_{}_final_{}_{}_{}.graph".format(
                                c, corrScale, distScale, norm)):
                    print("downSampled_4_Rotate_PI_{}_final_{}_{}_{}.graph not done yet".format(c, corrScale, distScale, norm))
                    results.append(
                        ["Rotate_PI_{}_final_{}_{}_{}".format(c, corrScale, distScale, norm), 0, 0,
                         0, 0])
                    continue
                endCenters = getCenters(
                    r"D:\Software\FlexibleRefinement\bin\Debug\downsampling\StepTwo\downSampled_2_Rotate_PI_{}_final_{}_{}_{}.graph".format(c,
                                                                                                                     corrScale,
                                                                                                                     distScale,
                                                                                                                     norm))
                displacements = endCenters - gtStartCenters
                offsets = endCenters - gtEndCenters
                offNorm = np.linalg.norm(offsets, axis=1)
                offMean = np.mean(offNorm)
                offStd = np.std(offNorm)
                dispDiff = displacements - gtDisplacements
                dispDiffNorm = np.linalg.norm(dispDiff, axis=1)
                dispDiffMean = np.mean(dispDiffNorm)
                dispDiffStd = np.std(dispDiffNorm)
                results.append(
                    ["downSampled_2_Rotate_PI_{}_final_{}_{}_{}".format(c, corrScale, distScale, norm), offMean, offStd, dispDiffMean,
                     dispDiffStd])
                # plt.hist(diff)
    df = pd.DataFrame(results, columns=["Filename", "Mean Pos Offset", "Std Pos Offset", "Diff of distance", "std"])
    df.to_csv('StepTwo.csv', index=False)
'''

if __name__ == "__main__":
    dosnwampledRotationExp()
    print("foo")
