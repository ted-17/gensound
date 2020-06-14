# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:11:10 2020

@author: Ted
"""
import os
import librosa
import numpy as np
from keras.layers import Input,Dense, Conv2D, MaxPooling2D, Flatten, UpSampling2D, Activation, BatchNormalization, Reshape
from keras.models import Model
from sklearn.model_selection import train_test_split
import time

import util
import network as nw

#%% make wavlist
dirname = "wav"
fftsize, hopsize, nbit = 512, 256, 8
wavlist = util.get_wavlist(dirname)
X = util.make_dataset(wavlist[1:], fftsize, hopsize, nbit)

#%% model training
height, width = fftsize//2, fftsize//2 #CNN height x width
X_train = X[:,:height,:width,...]
model,_ = nw.autoencoder(height, width)
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, X_train, epochs=5, batch_size=32)

#%% model testing
absY, phsY, max_Y, min_Y = util.make_spectrogram(wavlist[0], fftsize, hopsize, nbit, istest=True)
P = np.squeeze(model.predict(absY[np.newaxis,:height,:width,...]))
P = np.hstack((P,absY[:height,width:])) #t-axis
P = np.vstack((P,absY[height,:])) #f-axis
Y = (absY*(max_Y-min_Y)+min_Y)*phsY
y = librosa.core.istft(absY*phsY, hop_length=hopsize, win_length=fftsize)
