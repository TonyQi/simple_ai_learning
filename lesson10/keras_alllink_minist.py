import dataset
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import relu,softmax
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
from keras.utils import plot_model
import matplotlib.pyplot as plt

X_train, Y_train = dataset.get_minst_train_data_and_label()
X_test, Y_test = dataset.get_minst_test_data_and_label()

X_train = X_train.reshape(60000, 28*28)/255.0
X_test = X_test.reshape(10000, 28*28)/255.0

Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)


model = Sequential()
model.add(Dense(units=256, activation=relu,input_dim= 28*28))
model.add(Dense(units=256, activation=relu))
model.add(Dense(units=256, activation=relu))
model.add(Dense(units=10,activation=softmax))
model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])
plot_model(model, to_file='model.png')
history = model.fit(X_train, Y_train,epochs=100, batch_size=256)
print(history)

loss, accuracy = model.evaluate(X_test,Y_test)


# 绘制训练 & 验证的准确率值
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# # 绘制训练 & 验证的损失值
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

print('loss:%d; accuracy%d' %(loss, accuracy) )