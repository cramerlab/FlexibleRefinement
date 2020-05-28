#!/usr/bin/env python
import sys, os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn




df = pd.read_csv("pileupPart.tsv", '\t', sep='\t', names=["CONTIGID", "POS", "RBASE", "COVERAGE", "READBASES", "BASEQUAL"])
df2 = pd.read_csv("megahit.contigs.sorted.pileupPart", sep='\t', names=["CONTIGID", "POS", "RBASE", "COVERAGE", "READBASES", "BASEQUAL"] )
#print(list(df.columns.values))
bins=[0,2,5,10,20,50,100,500,1000,10000,max(df["COVERAGE"].max(), 100000)]
#plt.hist(df["CONTIGLEN"], bins=[0,1000,5000,10000,20000,40000,100000,df["CONTIGLEN"].max()])
#df=df.loc[df["TOOL"]=="PLASS-b8ea0ac"]
#print(df["CONTIGLEN"])

plt.figure()
ax = plt.gca()

num = len(df["TOOL"].unique())

totalWidth = 0.6

hist1 = np.histogram(df["COVERAGE"], bins=bins)
hist2 = np.histogram(df2["COVERAGE"], bins=bins)


ax.bar(np.arange(7) + totalWidth / num * 0 - (totalWidth / (2 * num)), hist1[0], totalWidth / num)
ax.bar(np.arange(7) + totalWidth / num * 1 - (totalWidth / (2 * num)), hist2[0], totalWidth / num)


ax.set_xticks(np.arange(len(bins)))
ax.set_xticklabels(["<2", "2-5", "5-10", "10-20", "20-50", "50-100","100-500", "500-1000", "1k-10k", ">10k"], rotation=45)
plt.ylabel("Coverage Value")
#plt.ylabel("number of circular contigs")
vals = ax.get_yticks()
plt.show()
