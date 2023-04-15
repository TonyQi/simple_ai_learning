import dataset
import plot_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import sigmoid
from keras.losses import mean_squared_error
from keras.optimizers import SGD
from keras.metrics import accuracy

m = 100
X, Y = dataset.get_beans4(m)

model = Sequential()
model.add(Dense(units=2, activation=sigmoid, input_dim=2))
model.add(Dense(units=1, activation=sigmoid))
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.05), metrics=['accuracy'])
model.fit(X, Y, epochs=10000, batch_size=10)

plot_utils.show_scatter_surface_with_model(X, Y, model)
