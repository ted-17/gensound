import os
import librosa
import numpy as np
from keras.layers import Input,Dense, Conv2D, MaxPooling2D, Flatten, UpSampling2D, Activation, BatchNormalization, Reshape
from keras.models import Model
import time
#file example

#%% AutoEncoderModel
def encoder(input_img):
    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    return encoded

def bottleneck(encoded):
    flt = Flatten()(encoded)
    dlen = int(np.prod(encoded.shape[1:]))
    btlnk = Dense(10, activation='relu')(flt)
    x = Dense(dlen, activation='relu')(btlnk)
    y = Reshape((int(encoded.shape[1]),int(encoded.shape[2]),int(encoded.shape[3])))(x) #correct??
    return y, btlnk
    
def decoder(input_img,encoded):
    x = Conv2D(16, (3, 3), padding='same')(encoded)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Activation('sigmoid')(x)
    model = Model(input_img, decoded)
    return model

def autoencoder(height, width):
    input_img = Input(shape=(height, width, 1))
    encoded = encoder(input_img)
    bn, btlnk = bottleneck(encoded)
    autoencoded = decoder(input_img, bn)
    return autoencoded, btlnk