import matplotlib.pyplot as plt
import os
import sys
import numpy as np


def plot1():
    x = np.arange(0, 10, 0.001, np.double)

    y1 = np.exp(-((x-4)**2/1.1))
    y2 = np.exp(-((x-6)**2/1.1))
    yGes = y1+y2

    atomsX = [4, 6]
    atomsY = [0.95, 0.95]


    fig, ax = plt.subplots()
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.scatter(atomsX, atomsY, s=80, color='r', label="atom positions")

    plt.plot(x, y1,'r--', )


    plt.plot(x, y2,'r--')

    plt.plot(x, yGes, label="represented density")

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="large")
    plt.tight_layout()
    #plt.show()
    fig.savefig(os.path.join(basedir, "atomRepr.png"), bbox_inches='tight')


basedir= r"D:\ownCloud\20191017_ProgressReport"
plot1()
x = np.arange(0, 12, 0.001, np.double)

Ay1 = np.exp(-((x - 2) ** 2 / 1.1))
Ay2 = np.exp(-((x - 4) ** 2 / 1.1))
AyGes = Ay1 + Ay2
AyGt = np.minimum(1.0, 1.5 * np.exp(-((x - 3) ** 2 / 2.5)))
AatomsX = [2, 4]
AatomsY = [0.95, 0.95]


By1 = np.exp(-((x - 8) ** 2 / 1.1))
By2 = np.exp(-((x - 10) ** 2 / 1.1))
ByGes = By1 + By2
ByGt = np.minimum(1.0, 1.5 * np.exp(-((x - 9) ** 2 / 2.5)))
BatomsX = [8, 10]
BatomsY = [0.95, 0.95]


fig, ax = plt.subplots()
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)

ax.plot(x, AyGt, "--", label="density A", color="grey")
plt.scatter(AatomsX, AatomsY, s=80, color='r', label="atom positions")
plt.plot(x, AyGes, label="represented density")

#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="large")
plt.tight_layout()
fig.savefig(os.path.join(basedir, "Density_A_withAtoms.png"), bbox_inches='tight')


fig, ax = plt.subplots()
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
ax.plot(x, ByGt, "--", label="density A", color="grey")
plt.scatter(BatomsX, BatomsY, s=80, color='r', label="atom positions")
plt.plot(x, ByGes, label="represented density")
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="large")
plt.tight_layout()
fig.savefig(os.path.join(basedir, "Density_B_withAtoms.png"), bbox_inches='tight')

fig, ax = plt.subplots()
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
ax.plot(x, AyGt, "--", label="density A", color="grey")
ax.plot(x, ByGt, "--", label="density B", color="blue")

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="large")
plt.tight_layout()
fig.savefig(os.path.join(basedir, "Density_AandB.png"), bbox_inches='tight')


plt.plot(x, Ay1, 'r--')
plt.plot(x, Ay2, 'r--')
plt.scatter(AatomsX, AatomsY, s=80, color='r', label="atom positions")
plt.plot(x, AyGes, label="represented density")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="large")
fig.savefig(os.path.join(basedir, "Density_AandB_atomsStart.png"), bbox_inches='tight')


'''
fig, ax = plt.subplots()

plt.scatter(AatomsX, AatomsY, s=80, color='r', label="atom positions")

plt.plot(x, Ay1, 'r--' )

plt.plot(x, Ay2, 'r--')

plt.plot(x, AyGt, "--", label="density", color="grey")

plt.plot(x, AyGes, label="represented intensity")

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
fig.savefig(os.path.join(basedir, "Density_AandB_2.png"), bbox_inches='tight')'''