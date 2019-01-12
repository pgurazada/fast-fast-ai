import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential

# Read and save data
data = open('amazon-reviews-corpus').read()
labels, texts = [], []

for i, line in enumerate(data.split('\n')):
    content = line.split()
    labels.append(content[0])
    texts.append(" ".join(content[1:]))

train_df = pd.DataFrame()
train_df['text'] = texts
train_df['label'] = labels
train_df.to_feather('amazon-reviews-corpus.feather')


target_encoder = LabelEncoder()
train_y = target_encoder.fit_transform(train_df['label'])
train_X = train_df['text']

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(train_X)

sequences = tokenizer.texts_to_sequences(train_X)

data = pad_sequences(sequences, maxlen=300)

model = Sequential()
model.add(Embedding(20000, 128, input_length=300))
model.add(LSTM(128, dropout=.2, recurrent_dropout=.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(data, to_categorical(train_y), 
          validation_split=.2, 
          epochs=3)
