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
        lines = lines[2:]
        centers = np.array([[float(f) for f in line.split()[1:4]] for line in lines])
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
                fig = plt.figure()
                plt.hist(displacements)
                plt.savefig(
                    r"D:\Software\FlexibleRefinement\bin\Debug\RotateStick\Rotate_PI_{}_final_{}_{}_{}.png".format(c,
                                                                                                                   corrScale,
                                                                                                                   distScale,
                                                                                                                   norm))
                plt.close(fig)
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


def plotIterationDiff(step="StepThree", c=10):
    basepath = r"D:\Software\FlexibleRefinement\bin\Debug\lennardJones\GridSearchParams_No_NeighborUpdate_c10000"
    if step == "StepOne":
        factor = 4
    elif step == "StepTwo":
        factor = 2
    elif step == "StepThree":
        factor = 1
    numIt = 500
    gtStartCenters = getCenters(basepath + "\{}_StartGraph.graph".format(factor))
    corrScale = 1
    distScale = 7
    norm = "False"
    displMatrix = np.zeros((numIt, gtStartCenters.shape[0]))
    prev = gtStartCenters
    for it in range(1, numIt+1):
        now = getCenters(basepath + "\{}\{}_Rotate_PI_{}_{}_{}_{}_it{}.graph".format(
            step, factor, c, corrScale, distScale, norm, it))
        dispalcements = np.linalg.norm(now-prev, axis=1)
        displMatrix[it-1,:] = np.linalg.norm(now-prev, axis=1)
        prev = now
    print("done")

def downsampledRotationExp(step="StepThree", c=10):
    basepath = r"D:\Software\FlexibleRefinement\bin\Debug\PulledProtein\empiar_10216\trial1"

    if step == "StepOne":
        factor = 4
    elif step == "StepTwo":
        factor = 2
    elif step == "StepThree":
        factor = 1
    gtStartCenters = getCenters(basepath + "\{}_StartGraph.graph".format(factor))
    gtEndCenters = getCenters(basepath + "\{}_{}_TarGraph.graph".format(factor, c))
    gtDisplacements = gtEndCenters - gtStartCenters
    results = []
    done = 0
    notDone = 0
    for corrScale in range(1, 21):
        for distScale in range(1, 21):
            for norm in ["True", "False"]:
                if not os.path.isfile(
                        basepath + "\{}\{}_Rotate_PI_{}_{}_{}_{}_final.graph".format(
                                step, factor, c, corrScale, distScale, norm)):
                    print("{}_Rotate_PI_{}_{}_{}_{}_final.graph not done yet".format(factor, c, corrScale, distScale,
                                                                                     norm))
                    results.append(
                        ["{}_Rotate_PI_{}_{}_{}_{}".format(factor, c, corrScale, distScale, norm), 0, 0,
                         0, 0])
                    notDone += 1
                    continue
                endCenters = getCenters(
                    basepath + "\{}\{}_Rotate_PI_{}_{}_{}_{}_final.graph".format(
                        step,
                        factor,
                        c,
                        corrScale,
                        distScale,
                        norm))
                done += 1
                displacements = endCenters - gtStartCenters
                #fig = plt.figure()
                #plt.hist(np.linalg.norm(displacements, axis=1))
                #plt.savefig(basepath + "\{}\{}_Rotate_PI_{}_{}_{}_{}_final.png".format(
                #        step,
                #        factor,
                #        c,
                #        corrScale,
                #        distScale,
                #        norm))
                #plt.close(fig)
                offsets = endCenters - gtEndCenters
                offNorm = np.linalg.norm(offsets, axis=1)
                offMean = np.sqrt(np.mean(offNorm**2))
                offStd = np.std(offNorm)
                dispDiff = displacements - gtDisplacements
                dispDiffNorm = np.linalg.norm(dispDiff, axis=1)
                dispDiffMean = np.mean(dispDiffNorm)
                dispDiffStd = np.std(dispDiffNorm)
                results.append(
                    ["{}_Rotate_PI_{}_{}_{}_{}".format(factor, c, corrScale, distScale, norm), offMean, offStd,
                     dispDiffMean, dispDiffStd])
                # plt.hist(diff)
    df = pd.DataFrame(results, columns=["Filename", "Mean Pos Offset", "Std Pos Offset", "Diff of distance", "std"])
    df.to_csv(basepath + r'\c_{}_{}.csv'.format(c, step), index=False)
    print("Done: {} ({} %) NotDone: {} ({} %)".format(done, done/(done+notDone)*100, notDone,
                                                      notDone/(done+notDone)*100))
    '''
    gtStartCenters = getCenters(basepath + "\StartGraph_downSampled_2.graph")
    gtEndCenters = getCenters(basepath + "\TargetGraph_downSampled_2.graph")
    gtDisplacements = gtEndCenters - gtStartCenters
    results = []
    for corrScale in range(1, 21):
        for distScale in range(1, 21):
            for norm in ["True", "False"]:
                if not os.path.isfile(
                        basepath + "\StepTwo\downSampled_2_Rotate_PI_{}_final_{}_{}_{}.graph".format(
                                c, corrScale, distScale, norm)):
                    print("downSampled_4_Rotate_PI_{}_final_{}_{}_{}.graph not done yet".format(c, corrScale, distScale,
                                                                                                norm))
                    results.append(
                        ["Rotate_PI_{}_final_{}_{}_{}".format(c, corrScale, distScale, norm), 0, 0,
                         0, 0])
                    continue
                endCenters = getCenters(
                    basepath + "\StepTwo\downSampled_2_Rotate_PI_{}_final_{}_{}_{}.graph".format(c,
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
                    ["downSampled_2_Rotate_PI_{}_final_{}_{}_{}".format(c, corrScale, distScale, norm), offMean, offStd,
                     dispDiffMean, dispDiffStd])
                # plt.hist(diff)
    df = pd.DataFrame(results, columns=["Filename", "Mean Pos Offset", "Std Pos Offset", "Diff of distance", "std"])
    df.to_csv('StepTwo.csv', index=False)
'''

def forceFieldExp(step="StepThree"):
    basepath = r"D:\Software\FlexibleRefinement\bin\Debug\PulledProtein\Toy\100\current_trial10000"

    if step == "StepOne":
        factor = 4
    elif step == "StepTwo":
        factor = 2
    elif step == "StepThree":
        factor = 1
    gtStartCenters = getCenters(basepath + "\{}_StartGraph.xyz".format(factor))
    gtEndCenters = getCenters(basepath + "\{}_TargetGraph.xyz".format(factor))
    gtDisplacements = gtEndCenters - gtStartCenters
    results = []
    done = 0
    notDone = 0
    for corrScale in range(1, 21):
        for distScale in range(1, 21):
            for norm in ["True", "False"]:
                if not os.path.isfile(
                        basepath + "\{}\{}_Rotate_PI_{}_{}_{}_final.xyz".format(
                                step, factor, corrScale, distScale, norm)):
                    print("{}_Rotate_PI_{}_{}_{}_final.graph not done yet".format(factor, corrScale, distScale,
                                                                                     norm))
                    results.append(
                        ["{}_Rotate_PI_{}_{}_{}".format(factor, corrScale, distScale, norm), 0, 0,
                         0, 0])
                    notDone += 1
                    continue
                endCenters = getCenters(
                    basepath + "\{}\{}_Rotate_PI_{}_{}_{}_final.xyz".format(
                        step,
                        factor,
                        corrScale,
                        distScale,
                        norm))
                done += 1
                displacements = endCenters - gtStartCenters
                #fig = plt.figure()
                #plt.hist(np.linalg.norm(displacements, axis=1))
                #plt.savefig(basepath + "\{}\{}_Rotate_PI_{}_{}_{}_{}_final.png".format(
                #        step,
                #        factor,
                #        c,
                #        corrScale,
                #        distScale,
                #        norm))
                #plt.close(fig)
                offsets = endCenters - gtEndCenters
                offNorm = np.linalg.norm(offsets, axis=1)
                offMean = np.sqrt(np.mean(offNorm**2))
                offStd = np.std(offNorm)
                dispDiff = displacements - gtDisplacements
                dispDiffNorm = np.linalg.norm(dispDiff, axis=1)
                dispDiffMean = np.mean(dispDiffNorm)
                dispDiffStd = np.std(dispDiffNorm)
                results.append(
                    ["{}_Rotate_PI_{}_{}_{}".format(factor, corrScale, distScale, norm), offMean, offStd,
                     dispDiffMean, dispDiffStd])
                # plt.hist(diff)
    df = pd.DataFrame(results, columns=["Filename", "Mean Pos Offset", "Std Pos Offset", "Diff of distance", "std"])
    df.to_csv(basepath + r'\{}.csv'.format(step), index=False)
    print("Done: {} ({} %) NotDone: {} ({} %)".format(done, done/(done+notDone)*100, notDone,
                                                      notDone/(done+notDone)*100))

if __name__ == "__main__":
    for s in ['StepOne', 'StepTwo', 'StepThree']:
        forceFieldExp(step=s)
    #downsampledRotationExp(step="StepThree", c=8)
    print("foo")
