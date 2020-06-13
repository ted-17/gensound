# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:11:10 2020

@author: Ted
"""
import numpy as np
import librosa
import glob
#%% utility methods
def get_wavlist(dir):
    """
    find .wav file and make wavlist from the specified dir
    """
    get_list = glob.glob(dir)
    return get_list

def make_spectrogram(pathname, fftsize, hopsize, nbit, istest=False):
    """
    make spectrogram from pathname
    making it easy for network to train data, compress-normalize with "nbit"
    """
    x,_ = librosa.core.load(pathname, sr=16000) #audioread
    X = librosa.core.stft(x, n_fft=fftsize, hop_length=hopsize)
    absX = np.abs(X)
    phsX = np.exp(1.j*np.angle(X))
    maxval, minval = np.max(absX), np.min(absX)
    absXn_int= np.floor(((absX-minval)/(maxval-minval))*(2**nbit-1) + 0.5)
    absXn = absXn_int/(2**nbit-1) #0-1
    if istest:
        return absXn, phsX, maxval, minval
    else:
        return absXn, phsX

def make_dataset(wavlist, fftsize, hopsize, nbit):
    """
    find .wav file and make spectrogram
    """
    Xbox = []
    for path in wavlist:
        X,_ = make_spectrogram(path, fftsize, hopsize, nbit)
        Xbox.append(X)
    return np.array(Xbox)[...,np.newaxis]