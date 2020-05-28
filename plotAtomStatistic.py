import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\EMD\9233\tmp\fscs.tsv", sep='\t')

df_tmp = df.loc[df["ROUND"]==1]
sn.lineplot("ATOMCOUNT", "RESOLUTION", hue="OVERSAMPLING", data=df_tmp)
plt.show()
