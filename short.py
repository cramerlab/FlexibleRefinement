import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = {'Megahit' : np.random.normal(loc = 0, size = 10),
        'plass' : np.random.normal(loc = 0, size = 10),
        'location' : ['illinois', 'illinois', 'illinois', 'Japan', 'Japan', 'Japan', 'Singapur', 'Singapur', 'Singapur', 'Singapur'],
        'sample': np.array(range(1,11))}

df = pd.DataFrame(data)
print(df)
df.boxplot(by='location')
plt.show()
'''
from glob import glob

for i, f in enumerate(glob(r"D:\EMD\9233\**.fsc")):
    f = f.replace('\\', '\\\\')
    if not f.endswith("masked.fsc"):
        print('MRCImage<DOUBLE> tmp{} = MRCImage<DOUBLE>::readAs("{}");'.format(i, f.replace(".fsc",".mrc")))
        print('writeFSC(origMasked(), tmp{}()*Mask(), "{}");'.format(i,f.replace(".fsc", "_masked.fsc")))
        print('tmp{}.writeAs<float>("{}");'.format(i, f.replace(".fsc", "_masked.mrc")))'''