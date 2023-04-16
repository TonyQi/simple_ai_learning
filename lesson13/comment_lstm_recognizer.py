import shopping_data
import chinese_vec
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Embedding, Dense, LSTM
from keras.activations import sigmoid
from keras.optimizers import adam
from keras.metrics import accuracy
from keras.losses import binary_crossentropy
import numpy as np

X_train, Y_train, X_test, Y_test = shopping_data.load_data()

vocalen, word_index = shopping_data.createWordIndex(X_train, X_test)

X_train = shopping_data.word2Index(X_train, word_index)
X_test = shopping_data.word2Index(X_test, word_index)
maxlength = 250
X_train = sequence.pad_sequences(X_train, maxlen=maxlength)
X_test = sequence.pad_sequences(X_test, maxlen=maxlength)

chinese_vess = chinese_vec.load_word_vecs()

embedings_vectors = np.zeros((vocalen, 300))
for word,i in word_index.items():
    embedings_vector = chinese_vess.get(word)
    if embedings_vector is not None:
        embedings_vectors[i] = embedings_vector
print(embedings_vectors)
model = Sequential()
model.add(
    Embedding(trainable=False, input_dim=vocalen, output_dim=300, weights=[embedings_vectors], input_length=maxlength))
model.add(LSTM(units=128, return_sequences=True))
model.add(LSTM(units=128))
model.add(Dense(units=1, activation=sigmoid))
model.compile(loss='binary_crossentropy', optimizer=adam(), metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=500, batch_size=128)
loss, accuracy = model.evaluate(X_test, Y_test)
print("loss is ",loss )
print("accuracy is ",accuracy )
model.save('comment_lstm_model.h5')
