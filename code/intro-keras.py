'''
In this script we use a basic keras model to explore the impact of varying the
optimizers and the learning rate

The model is kept very simple and the data set is MNIST
'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import logging

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, Adadelta, RMSprop

# logging settings

logging.basicConfig(filename='accuracy-trace.log', 
                    filemode='w',
                    level=logging.INFO)


# plot settings
sns.set_context('talk')
sns.set_style('ticks')

# data exploration
digits = load_digits()
sample_index = 45

data = np.array(digits.data, dtype='float32')
target = np.array(digits.target, dtype='int32')

X_train, X_test, y_train, y_test = train_test_split(data, target, 
                                                    test_size=.2, 
                                                    random_state=20130810)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Before standardization
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(digits.images[sample_index],
           cmap=plt.cm.gray_r,
           interpolation='nearest')
plt.title('image label: %d' % digits.target[sample_index])

# After standardization
plt.subplot(122)
plt.imshow(X_train[sample_index].reshape(8, 8),
           cmap=plt.cm.gray_r,
           interpolation='nearest')
plt.title('transformed image (label: %d)' % digits.target[sample_index])

plt.show()

y_train = to_categorical(y_train)
learning_rate = 0.1
for learning_rate in [0.1, 0.01, 0.001, 0.0001]:

    for optimizer_obj in [SGD(lr=learning_rate), Adam(lr=learning_rate),
                        Adadelta(lr=learning_rate), RMSprop(lr=learning_rate)]:
        
        model_basic = Sequential()

        model_basic.add(Dense(100, input_dim=X_train.shape[1]))
        model_basic.add(Activation('relu'))

        model_basic.add(Dense(10))
        model_basic.add(Activation('softmax'))

        model_basic.compile(optimizer=optimizer_obj,
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

        model_basic.fit(X_train, y_train, epochs=15, batch_size=32)

        y_preds = model_basic.predict_classes(X_test)

        logging.info(str(optimizer_obj) + 'learning rate: %.4f, test accuracy: %.3f' % 
                    (learning_rate, (y_preds == y_test).mean()))



        
        
