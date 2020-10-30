import pandas
import matplotlib.pyplot as plt

df = pandas.read_csv(r"AtomFitting\iterationErr.txt",  header=None, names=["err"])
df["err"]=df["err"].astype(float)
df.plot()

plt.show()
