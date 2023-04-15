import dataset
import plot_utils
import numpy as np

m = 100
xs, ys = dataset.get_beans(100)
print(xs, ys)
plot_utils.show_scatter(xs, ys)

w1 = np.random.rand()
w2 = np.random.rand()
b = np.random.rand()


def sigmod(z):
    e = 1 / (1 + np.exp(-z))
    return e


x1s = xs[:, 0]
x2s = xs[:, 1]


def forward_propagate(x1, x2):
    a = w1 * x1 + w2 * x2 + b
    z = sigmod(a)
    return z


plot_utils.show_scatter_surface(xs, ys, forward_propagate)

for _ in range(500):
    for i in range(m):
        x = xs[i]
        y = ys[i]
        x1 = x[0]
        x2 = x[1]
        z = forward_propagate(x1, x2)
        e = (y - z) ** 2
        dedz = 2*(y - z) * -1
        dzda = z * (1 - z)
        dadw1 = x1
        dadw2 = x2
        dadb = 1
        dedw1 = dedz * dzda * dadw1
        dedw2 = dedz * dzda * dadw2
        dedb = dedz * dzda * dadb
        w1 = w1 - dedw1
        w2 = w2 - dedw2
        b = b - dedb
plot_utils.show_scatter_surface(xs, ys, forward_propagate)
