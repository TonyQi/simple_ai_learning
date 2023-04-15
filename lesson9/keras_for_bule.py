import dataset
import plot_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import relu
from keras.activations import sigmoid
from keras.losses import mean_squared_error
from keras.optimizers import SGD
from keras.metrics import accuracy
m = 100
X, Y =dataset.get_beans(m)
plot_utils.show_3d_scatter(X,Y)

model = Sequential()
model.add(Dense(units=8,activation=relu,input_dim=2))
model.add(Dense(units=8,activation=relu))
model.add(Dense(units=8,activation=relu))
model.add(Dense(units=1,activation=sigmoid))

model.compile(loss='mean_squared_error',optimizer=SGD(lr=0.05),metrics=['accuracy'])
model.fit(X,Y,epochs=20000,batch_size=100)

plot_utils.show_scatter_surface_with_model(X,Y,model)