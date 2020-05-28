import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv(sys.argv[1], sep='\t', names=["ProteinId", "StartCoord", "EndCoord", "Strand", "Partial", "ContigId", "ContigLen", "ProteinName", "HMMScore"])
tmp = df.loc[df["Strand"] == -1]["EndCoord"]

df.loc[df["Strand"] == -1, "EndCoord"] = df.loc[df["Strand"] == -1, "ContigLen"] - df.loc[df["Strand"] == -1, "StartCoord"]+1

df.loc[df["Strand"] == -1, "StartCoord"] = df.loc[df["Strand"] == -1, "ContigLen"] - tmp + 1

df.loc[df["Strand"] == -1, "Strand"] = 1
maxlen = df["ContigLen"].max()

kernel = 1

RdRpPositions = np.zeros(maxlen//kernel)
MPPositions = np.zeros(maxlen//kernel)
CPPositions = np.zeros(maxlen//kernel)

for start, end in df.loc[df['ProteinName'].str.contains('CP'), ["StartCoord", "EndCoord"]].itertuples(index=False):
    CPPositions[start//kernel:(end+1)//kernel] += 1

for start, end in df.loc[df['ProteinName'].str.contains('MP'), ["StartCoord", "EndCoord"]].itertuples(index=False):
    MPPositions[start//kernel:(end+1)//kernel] += 1

for start, end in df.loc[df['ProteinName'].str.contains('RdRp'), ["StartCoord", "EndCoord"]].itertuples(index=False):
    RdRpPositions[start//kernel:(end+1)//kernel] += 1
    if(start>10000):
        pass
fig,axs = plt.subplots(2,1)

axs[0].plot(np.arange(1, maxlen+1,kernel), RdRpPositions, "r")
axs[0].plot(np.arange(1, maxlen+1,kernel),MPPositions, "b")
axs[0].plot(np.arange(1, maxlen+1,kernel),CPPositions, "g")
axs[0].set_yscale("log")
axs[1].boxplot(df["ContigLen"], 0, 's', 0)
plt.show()