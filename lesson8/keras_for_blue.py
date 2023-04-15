import dataset
import plot_utils
from keras.models import Sequential
from keras.layers import Dense

m = 100
X, Y = dataset.get_beans1(m)

model = Sequential()
model.add(Dense(units=1, activation='sigmoid', input_dim=1))

model.compile(loss='mean_squared_error', optimizer="sgd", metrics=['accuracy'])
model.fit(X, Y, epochs=50000, batch_size=10)

y_pre = model.predict(X)

plot_utils.show_scatter_curve(X, Y, y_pre)
