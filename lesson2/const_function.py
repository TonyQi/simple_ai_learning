import numpy

import dataset
import matplotlib.pyplot as plt
import numpy as np
m = 100
xs, ys = dataset.get_beans(m)

def variance(w, xs, ys):
    y_pre = w * xs
    e = np.sum((ys - y_pre) ** 2) / m
    return e

es = []

for w in np.arange(-5,5,0.1):
    e = variance(w, xs, ys)
    es.append(e)

plt.title("方差变化情况")
plt.xlabel("w")
plt.ylabel("e")
plt.scatter(np.arange(-5,5,0.1),es)
plt.show()








