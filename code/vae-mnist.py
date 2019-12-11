import numpy as np

from keras.layers import Input, Conv2D, Flatten, Dense, Lambda
from keras import backend as K

img_shape = (28, 28, 1)
batch_size = 16
latent_dim = 2  # dimensionality of the latent space

input_img = Input(shape=img_shape)

x = Conv2D(32, 3, padding='same', activation='relu')(input_img)
x = Conv2D(64, 3, pading='same', activation='relu', strides=(2, 2))(x)
x = Conv2D(64, 3, pading='same', activation='relu')(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)

shape_before_flattening = K.int_shape(x)

x = Flatten()(x)
x = Dense(32, activation='relu')(x)

z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

def sampling(args):

    '''
    Latent space sampling function

    '''
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0.0, stddev=1.0)

    return z_mean + K.exp(z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])
