import dataset
from matplotlib import pyplot as plot

i = 6
xs, ys = dataset.get_beans(i)
print(xs.size)


plot.title("BeanSize-Toxicity")
plot.xlabel("BeanSize")
plot.ylabel("Toxicity")
plot.scatter(xs, ys)

w = 200
a = 0.3

for mm in range(i):
    for m in range(i):
        y_prei = w * xs[m]
        w = w + (ys[m] - y_prei) * xs[m]
y_pre = w * xs
plot.plot(xs, y_pre)
plot.show()
