import os
import sys
import matplotlib.pyplot as plt
from glob import glob
import re
#infileDir = r"D:\EMD\9233\emd_9233_Scaled_1.5_75k_bs64_it3_moving"
infileDir = r"D:\EMD\9233\tmp"
voxelSize = 1.5


def findIntersect(a, x1,y1,x2,y2):

    return (a-(x2*y1-x1*y2)/(x2-x1))*(x2-x1)/(y2-y1)

for f in glob(os.path.join(infileDir, r"*.fsc"), recursive=True):

    pixels = []
    values = []
    error = False;
    found1 = False
    found5 = False
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

    fig = plt.figure()
    plt.margins(x=0, y=0)
    plt.plot(pixels, values)

    #plt.text(cutOff, -0.1, "{:.2f} A".format(1/cutOff), color="r")

    bottom = plt.gca().get_ylim()[0]
    bottom = min(bottom, 0.0)
    plt.gca().set_ylim([bottom, 1.1])
    # Draw cutoff lines
    xticks = []
    labels = []
    if found5:
        cutOff5 = cutOff5 * 1 / (boxsize * voxelSize)
        plt.text(cutOff5+0.01, 0.5 + 0.01, "{:.2f}Å".format( 1 / cutOff5))
        xticks.append(cutOff5)
        labels.append("{:.2f}\n(1/{:.2f}Å)".format(cutOff5, 1 / cutOff5))
        plt.plot([0, cutOff5], [0.5, 0.5], "r--")
        plt.stem([cutOff5], [0.5], "r--", bottom=bottom)
    #if found1:
    #    cutOff1 = cutOff1 * 1 / (boxsize * voxelSize)
    #    xticks.append(cutOff1)
    #    plt.text(cutOff1 + 0.01, 0.143 + 0.01, "{:.2f}Å".format(1 / cutOff1))
    #    labels.append("{:.2f}\n(1/{:.2f}Å)".format(cutOff1, 1 / cutOff1))
    #    plt.plot([0, cutOff1], [0.143, 0.143], "r--")
    #    plt.stem([cutOff1], [0.143], "r--", bottom=bottom)
    xticks.append(pixels[-1])
    labels.append("{:.2f}\n(1/{:.2f}Å)".format(pixels[-1], 1 / pixels[-1]))

    #plt.gca().set_xticks(xticks)
    #plt.gca().set_xticklabels(labels)


    plt.ylabel("FSC")
    plt.xlabel("Spatial Freq. [1/Å]")
    plt.savefig(f.replace(".fsc", ".png"), bbox_inches='tight')

