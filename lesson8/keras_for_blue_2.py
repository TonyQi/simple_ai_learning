import dataset
import plot_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

m = 100
X, Y = dataset.get_beans2(m)

model = Sequential()
model.add(Dense(units=5, activation='sigmoid', input_dim=1))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.05), metrics=['accuracy'])

model.fit(X, Y, epochs=5000, batch_size=10)

y_pre = model.predict(X)

plot_utils.show_scatter_curve(X, Y, y_pre)
