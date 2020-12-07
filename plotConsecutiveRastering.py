import numpy as np
import os
import seaborn
import matplotlib.pyplot as plt
from glob import glob
for dir in glob(r"D:\EMD\9233\Consecutive_Rastering\*"):
    indir = os.path.join(r"D:\EMD\9233\Consecutive_Rastering", dir)
    if not os.path.isdir((indir)):
        continue


    numit = 10


    fig_fsc, ax_fsc = plt.subplots()
    fig_frc, ax_frc = plt.subplots()

    for i in range(10):
        fscfilepath = os.path.join(indir, "{}_fsc.star".format(i))
        frcfilepath = os.path.join(indir, "{}_frc_vs_ref_masked.star".format(i))
        values = []

        with open(fscfilepath, "r") as fscfile:
            for line in fscfile:
                if line.startswith("#") or line.startswith("data") or line.startswith("loop") or line.startswith("_") or not line.strip():
                    continue
                else:
                    values.append(float(line.strip().split()[1]))
            ax_fsc.plot(values, label=str(i))
        values = []

        with open(frcfilepath, "r") as frcfile:
            for line in frcfile:
                if line.startswith("#") or line.startswith("data") or line.startswith("loop") or line.startswith("_") or not line.strip():
                    continue
                else:
                    values.append(float(line.strip()))
            ax_frc.plot(values, label=str(i))
    ax_fsc.set_yticks(np.arange(0, 1.1, step=0.1))
    ax_fsc.set_ylim((0, 1.1))

    ax_frc.set_yticks(np.arange(0, 1.1, step=0.1))
    ax_frc.set_ylim((0, 1.1))

    fig_fsc.savefig(os.path.join(indir, "fsc.png"))
    fig_frc.savefig(os.path.join(indir, "frc.png"))
    plt.close(fig_fsc)
    plt.close(fig_frc)