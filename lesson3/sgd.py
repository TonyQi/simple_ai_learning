import dataset
import matplotlib.pyplot as plt
import numpy as np
m = 100
xs, ys = dataset.get_beans(m)
#初始w
w = 0.1
#学习率
alpha = 0.1
plt.title("预测")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(xs,ys)
y_pre = w*xs
plt.plot(xs,y_pre)
plt.show()
for i in  range(100):
      k = (2 * np.sum(xs**2)*w -  2*np.sum(ys*xs))/100
      w = w - alpha * k
      y_pre = w * xs
      plt.clf()
      plt.xlim(0,1)
      plt.ylim(0,1)
      plt.plot(xs,y_pre)
      plt.scatter(xs,ys)
      plt.pause(0.01)
      plt.text(1,1,i)