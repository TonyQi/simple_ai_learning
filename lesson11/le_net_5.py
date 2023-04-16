from lesson10 import dataset
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
from keras.activations import relu, softmax
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
from keras.metrics import accuracy
from keras.utils import plot_model


X_train, Y_train = dataset.get_minst_train_data_and_label()
X_test, Y_test = dataset.get_minst_test_data_and_label()

X_train = X_train.reshape(60000, 28, 28, 1) / 255.0
X_test = X_test.reshape(10000, 28, 28, 1) / 255.0
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), input_shape=(28, 28, 1), padding='valid', activation=relu))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation=relu))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=120, activation=relu))
model.add(Dense(units=84, activation=relu))
model.add(Dense(units=10, activation=softmax))

model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])
plot_model(model, to_file='model.png')

model.fit(X_train, Y_train, epochs=500, batch_size=128)

loss,accuracy = model.evaluate(X_test, Y_test)
print('loss is %d, accuracy is %d' %(loss,accuracy))