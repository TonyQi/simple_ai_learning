import shopping_data
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten
from keras.activations import relu, sigmoid
from keras.losses import binary_crossentropy
from keras.optimizers import adam
from keras.metrics import accuracy

X_train, Y_train, X_test, Y_test = shopping_data.load_data()

vocalen, word_index = shopping_data.createWordIndex(X_train, X_test)

X_train = shopping_data.word2Index(X_train, word_index)
X_test = shopping_data.word2Index(X_test, word_index)
print(X_train.shape)
print(X_train[1])
maxlength = 256

X_train = sequence.pad_sequences(X_train, maxlen=maxlength)
X_test = sequence.pad_sequences(X_test, maxlen=maxlength)

print(X_train[1])

model = Sequential()
model.add(Embedding(trainable=True, input_dim=vocalen, output_dim=300, input_length=maxlength))
model.add(Flatten())
model.add(Dense(units=256, activation=relu))
model.add(Dense(units=128, activation=relu))
model.add(Dense(units=32, activation=relu))
model.add(Dense(units=1, activation=sigmoid))

model.compile(loss='binary_crossentropy', optimizer=adam(), metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=500, batch_size=256)

loss, accuracy = model.evaluate(X_test, Y_test)
print('loss is %d, accuracy is %d' % (loss, accuracy))
