import dataset
import numpy as np
import matplotlib.pyplot as plt

m = 100
xs, ys = dataset.get_beans(m)

w = 0.1
b = 0.1
z = w * xs + b
e = 1 / (1 + np.exp(-z))

plt.title("activation")
plt.xlabel("x")
plt.ylabel("y")

plt.scatter(xs, ys)
plt.plot(xs, e)

plt.show()
alpha = 0.1
for _ in range(5000):
    for i in range(m):
        x = xs[i]
        y = ys[i]
        a = w * x + b
        h = 1 / (1 + np.exp(-a))
        e = (y - h) ** 2
        dedh = 2*h - 2*y
        dhda = h*(1-h)
        dadw = x
        dadb = 1
        dedw = dedh *dhda *dadw
        dedb = dedh * dhda *dadb
        w = w -alpha * dedw
        b = b - alpha* dedb
    if _ %100 ==0:
        y_pre = 1/(1+np.exp(-w * xs - b))
        plt.clf()
        plt.title("w = " + str(w) + "  b=" + str(b))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.scatter(xs, ys)
        plt.plot(xs, y_pre)
        plt.pause(0.01)
