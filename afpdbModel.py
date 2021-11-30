# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 08:51:01 2018

@author: a) Hrishikesh S.
         b) Karthik A.N.
         c) Bhagyashree B.R.
Downloaded AFPDB using CygWin and saved locally
We can read the signals and even them
"""
# cd "Desktop/SECOND YEAR/RESEARCH"

import wfdb
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

"""Loading all the files"""
all_files = os.listdir('afpdb/')

files_dat = [f for f in glob.glob("afpdb/*.dat")]
files_hea = [f for f in glob.glob("afpdb/*.hea")]
files_qrs = [f for f in glob.glob("afpdb/*.qrs")]

files = []
for f in files_dat:
    files.append(f.replace(".dat", ""))
#print("All files:")
#print(files)
whole_amps = []
for i in files:
    w = wfdb.io.rdrecord(i)
    whole_amps.append(w)
    #wfdb.plot.plot_wfdb(w, return_fig = True)

print(whole_amps[0].sig_len)
"""Plot all signals"""
#wfdb.plot.plot_all_records("AFPDB/")

def extractECGsignals(signal):
    ecg0 = []
    ecg1 = []
    for i in signal.p_signal:
        ecg0.append(i[0])
        ecg1.append(i[1])
    return ecg0, ecg1

def extractSignalsUtil(all_signals):
    all_ecg0s = []
    all_ecg1s = []
    for i in all_signals:
        ecg0, ecg1 = extractECGsignals(i)
        all_ecg0s.append(ecg0)
        all_ecg1s.append(ecg1)
    return all_ecg0s, all_ecg1s

ecg0Signals, ecg1Signals = extractSignalsUtil(whole_amps)

def SignalDimensions(signal_array):
    for i in signal_array:
        print(len(i))

#SignalDimensions(ecg0Signals)
#SignalDimensions(ecg1Signals)

"""Padding"""
e0 = pad_sequences(ecg0Signals, maxlen=230400, dtype='float', padding='post', truncating='post', value=0)

"""dividing into training and test data"""
ecg0_train, ecg0_test = train_test_split(ecg0Signals, test_size=0.20, random_state=42)
e0_train, e0_test = train_test_split(ecg0Signals, test_size=0.20, random_state=42)

"""K-means Clustering"""
#not sure about how to do this
#km = KMeans(n_jobs=-1, n_init=20, verbose=1, n_clusters=300)
#km.fit(ecg0_train)

#predictionKmeans = km.predict(ecg0_test)

"""Autoencoder"""
import keras.backend as K
from time import time
from keras import callbacks
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Dense, Input
from keras.initializers import VarianceScaling
from keras.engine.topology import Layer, InputSpec

#from tensorflow.python.keras.layers import InputLayer, Conv1D, Dense, Flatten, MaxPool1D
#from tensorflow.python.keras.models import Sequential
inputNDarray = e0


inputNDarray = np.array(inputNDarray)
print(inputNDarray.shape)
# this is our input placeholder
input_model = Input(shape = (230400,))
#input_model = Input(input_shape = inputNDarray.shape[1:])
# "encoded" is the encoded representation of the input
encoded = Dense(units = 500, activation ='relu')(input_model)
encoded = Dense(units = 500, activation='relu')(encoded)
encoded = Dense(units = 2000, activation='relu')(encoded)
encoded = Dense(units = 10, activation='sigmoid')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(units = 2000, activation='relu')(encoded)
decoded = Dense(units = 500, activation='relu')(decoded)
decoded = Dense(units = 500, activation='relu')(decoded)
decoded = Dense(units = 230400)(decoded)
# this model maps an input to its reconstruction
autoencoder = Model(input_model, decoded)
autoencoder.summary()
#  this model maps an input to its encoded representation
#encoder = Model(input_img, encoded)
autoencoder.compile(optimizer='adam', loss='mse')
train_history = autoencoder.fit(e0, e0, epochs = 20, batch_size = 20, verbose = 0) #validation_data=(val_x, val_x))
