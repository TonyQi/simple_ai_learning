import dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

m = 100
xs, ys = dataset.get_beans(m)
# fig = plt.figure()
# ax = Axes3D(fig)

w = 0.1
b = 0.1
alpha = 0.1

def variance(w, b, xs, ys):
    y_pre = w * xs + b
    e = np.sum((ys - y_pre) ** 2) / m
    return e
# plt.title("预测")
plt.xlabel("w")
plt.ylabel("e")
plt.xlim(0,1)
plt.ylim(0,1)
plt.scatter(xs,ys)
y_pre = w*xs +b
plt.plot(xs,y_pre)
plt.scatter(xs,ys)

plt.show()
# bs =  np.arange(-5,5,0.1)
# ws = np.arange(-5,5,0.1)
# for b in bs:
#     es = []
#     for w in ws:
#         e= variance(w, b, xs,ys)
#         es.append(e)
#     figure =ax.plot(ws,es,bs,zdir="y")
# plt.show()
for _ in  range(100):
    for i in range(m):
        x = xs[i]
        y = ys[i]
        dw = 2 * x**2 * w + 2 * x * b - 2*y*x
        db = 2*b+2 * w * x - 2 * y
        w = w - alpha * dw
        b = b - alpha * db
    y_pre = w*xs +b
    plt.clf()
    plt.title("w = " + str(w)+"  b="+ str(b))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.scatter(xs, ys)
    plt.plot(xs,y_pre)
    plt.pause(0.01)

print(w, b)


