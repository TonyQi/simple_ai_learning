import dataset
import numpy as np
import matplotlib.pyplot as plt


def sigmod(x):
    return 1 / (1 + np.exp(-x))


m = 100
xs, ys = dataset.get_beans(100)
plt.scatter(xs, ys)
w11_1 = np.random.rand()
b11_1 = np.random.rand()

w21_1 = np.random.rand()
b21_1 = np.random.rand()

w11_2 = np.random.rand()
w12_2 = np.random.rand()
b11_2 = np.random.rand()


def forward_disseminate(x):
    y1_1 = w11_1 * x + b11_1
    z1_1 = sigmod(y1_1)
    y2_1 = w21_1 * x + b21_1
    z2_1 = sigmod(y2_1)
    y1_2 = w11_2 * z1_1 + w12_2 * z2_1 + b11_2
    z1_2 = sigmod(y1_2)
    return y1_1, z1_1, y2_1, z2_1, y1_2, z1_2


y1_1, z1_1, y2_1, z2_1, y1_2, z1_2 = forward_disseminate(xs)
alpha = 0.01
plt.plot(xs, z1_2)
plt.show()
for _ in range(50000):
    for i in range(m):
        x = xs[i]
        y = ys[i]
        y1_1, z1_1, y2_1, z2_1, y1_2, z1_2 = forward_disseminate(x)

        dy1_1dw11_1 = x
        dy1_1db11_1 = 1
        dz1_1dy1_1 = z1_1 * (1 - z1_1)

        dy2_1dw21_1 = x
        dy2_1db21_1 = 1
        dz2_1dy2_1 = z2_1 * (1 - z2_1)

        dy1_2dw11_2 = z1_1
        dy1_2dz1_1 = w11_2
        dy1_2dw12_2 = z2_1
        dy1_2dz2_1 = w12_2
        dy1_2db11_2 = 1
        dz1_2dy1_2 = z1_2 * (1 - z1_2)
        dedz1_2 = -2 * y + 2 * z1_2

        dedw11_2 = dedz1_2 * dz1_2dy1_2 * dy1_2dw11_2
        dedw12_2 = dedz1_2 * dz1_2dy1_2 * dy1_2dw12_2
        dedb11_2 = dedz1_2 * dz1_2dy1_2 * dy1_2db11_2

        dedw11_1 = dedz1_2 * dz1_2dy1_2 * dy1_2dz1_1 * dz1_1dy1_1 * dy1_1dw11_1
        dedb11_1 = dedz1_2 * dz1_2dy1_2 * dy1_2dz1_1 * dz1_1dy1_1 * dy1_1db11_1
        dedw21_1 = dedz1_2 * dz1_2dy1_2 * dy1_2dz2_1 * dz2_1dy2_1 * dy2_1dw21_1
        dedb21_1 = dedz1_2 * dz1_2dy1_2 * dy1_2dz2_1 * dz2_1dy2_1 * dy2_1db21_1

        w11_1 = w11_2 - alpha * dedw11_1
        b11_1 = b11_1 - alpha * dedb11_1
        w21_1 = w21_1 - alpha * dedw21_1
        b21_1 = b21_1 - alpha * dedb21_1
        w11_2 = w11_2 - alpha * dedw11_2
        w12_2 = w12_2 - alpha * dedw12_2
        b11_2 = b11_2 - alpha * dedb11_2

    if _ % 1000 == 0:
        plt.clf()
        plt.scatter(xs, ys)
        print(w11_1,b11_1,w21_1,b21_1,w11_2,w12_2,b11_2)
        plt.title(str(_))
        y1_1, z1_1, y2_1, z2_1,y1_2, z1_2 = forward_disseminate(xs)
        plt.plot(xs, z1_2)
        plt.pause(0.01)
