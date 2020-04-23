import numpy as np
import matplotlib.pyplot as plt


n=11

for i in range(1,n):
    a = 65
    for j in range(1,n-i):
        print(end=' ')
    for j in range(1,i+1):
        print(j,end='')
    for j in range(i-1,0,-1):
        ch=chr(a+j-1)
        print(ch,end='')

    print()


inStar = r"D:\EMPIAR\10168\emd_4180_res7.projections.star"
column1 = 12
name1 = "Rotation"
column2 = 13
name2 = "Tilt"
column3 = 14
name3 = "Psi"

values1 = []
values2 = []
values3 = []

with open(inStar, "r") as starFile:
    for line in starFile:
        if not line.strip() or line.startswith("data_") or line.startswith("_") or line.startswith("loop_"):
            continue
        line = line.strip()
        splitted = line.split()
        if len(values1)==10000:
            break
        value1 = float(splitted[column1])
        values1.append(value1)
        value2 = float(splitted[column2])
        values2.append(value2)
        value3 = float(splitted[column3])
        values3.append(value3)

plt.scatter(values1, values3)
plt.show()