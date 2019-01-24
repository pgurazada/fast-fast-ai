from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPool1D, Dense
from keras.models import Sequential
from keras.callbacks import TensorBoard

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

max_features, max_len = 2000, 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

model = Sequential()

model.add(Embedding(max_features, 128,
                    input_length=max_len,
                    name='embedding_layer'))
model.add(Conv1D(32, 7, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(32, 7, activation='relu'))
model.add(GlobalMaxPool1D())

model.add(Dense(1))

model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128,
          validation_split=0.2,
          callbacks=[TensorBoard(log_dir='my_log_dir',
                                 histogram_freq=1,
                                 embeddings_freq=1)])
