# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:16:16 2019

@author: Hrishikesh S.
"""
# cd "Desktop/SECOND YEAR/RESEARCH"

import librosa
import numpy as np
import pandas as pd
from librosa import display
import matplotlib.pyplot as plt
import tensorflow.python.keras.callbacks
import keras.callbacks
import scipy
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import InputLayer, Conv1D, Dense, Flatten, MaxPool1D
from tensorflow.python.keras.models import Sequential

def data_extraction(folder_name, tmp):
    data_y = []
    spectrogram_list = []
    mfsc_list = []
    for j in range(tmp.shape[0]):
        for i in range(1, tmp.shape[1] - 1):
            #try: 
            data, sampling_rate = librosa.load(folder_name + tmp.iloc[j, 0].split('.')[0] +'.wav' )
            print(sampling_rate)
            """we are taking value at every 25ms, so to decide size of the array, from which we
            take the mfsc"""
            size_factor = 25*(sampling_rate//1000)
            temp_label = tmp.iloc[:, i].name.split('.')[0]
            print(temp_label)
            start = int(tmp.iloc[j, i])
            end = int(tmp.iloc[j, i+1])
            """extracting features"""
            while(start<=end):
                temp_data = data[start:start+size_factor]
                start = start + size_factor
                spec = librosa.feature.spectral_centroid(y=temp_data, sr=sampling_rate)
                mfscs = librosa.feature.melspectrogram(y=temp_data, sr=sampling_rate)
                plt.title(temp_label)
                display.waveplot(spec, sr=sampling_rate)
                plt.savefig("Aplots/Spectrogram/"+str(tmp.iloc[j, 0].split('.')[0])+"S"+str(start-size_factor)+"E"+str(start)+".png")
                plt.close()
                plt.title(temp_label)
                display.waveplot(mfscs, sr=sampling_rate)
                plt.savefig("Aplots/Melspectrogram/"+str(tmp.iloc[j, 0].split('.')[0])+"S"+str(start-size_factor)+"E"+str(start)+".png")
                plt.close()
                spectrogram_list.append(spec)
                mfsc_list.append(mfscs)
                data_y.append(temp_label)
                print("Status : Success!")
            #except:
            #print("Status : Failed!")
            #pass
    return spectrogram_list, mfsc_list, data_y;

def data_preprocessing(feature, data_y):
    """padding (just in case)"""
    feature = pad_sequences(feature, maxlen=128, dtype='float', padding='post', truncating='post', value=0.)
    """ normalizing data """
    feature = feature / np.max(feature)
    """assigning integral labels"""
    data_y = pd.Series(data_y)
    data_y.value_counts()
    data_y = data_y.map({'S1':0, 'S2':1}).values
    return feature, data_y;

def createMFCCModel(model, feature):
    print(np.shape(feature[1:2]))
    model.add(InputLayer(input_shape = np.shape(feature[1:])))
    """Hidden Layer 1"""
    model.add(Conv1D(filters=1, kernel_size=1, activation='sigmoid'))
    #model.add(MaxPool1D(strides=0))
    """Flattening and Output"""
    model.add(Flatten())
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','mae'])
    """ model architecture """
    model.summary()
    return model

"""MFCC training model"""
def modelMFCCTraining(model, feature_x, labels_y):
    model.fit(feature_x, labels_y, batch_size=100, epochs=10)
    return model;

"""DATASET B MFSC and spectrogram"""
spec_list, mfsc_list, data_yb = data_extraction("Training B Normal/", pd.read_csv('Btraining_normal_seg.csv'))
mfsc_list_pos, data_yb = data_preprocessing(mfsc_list, data_yb)
spec_list_pos, data_yb_pos = data_preprocessing(spec_list, data_yb)
modelb = Sequential()
modelb = createMFCCModel(modelb, spec_list_pos)
modelb = modelMFCCTraining(modelb, mfsc_list_pos, data_yb)

"""DATASET A MFSC and spectrogram"""
Aspec_list, Amfsc_list_original, data_ya = data_extraction("Atraining_normal/", pd.read_csv('Atraining_normal_seg.csv'))
Amfsc_list= []
Amfsc_list_pos, data_ya = data_preprocessing(Amfsc_list, data_ya)
Aspec_list_pos, data_ya_pos = data_preprocessing(Aspec_list, data_ya)
modela = Sequential()
modela = createMFCCModel(modela, Aspec_list_pos)
modela = modelMFCCTraining(modela, mfsc_list_pos, data_ya)

del Amfsc_list[305] 
del data_yb[305]
del Amfsc_list[539] 
del data_yb[539]
del Amfsc_list[856] 
del data_yb[856]
del Amfsc_list[1191] 
del data_yb[1191]
del Amfsc_list[1870] 
del data_yb[1870]
del Amfsc_list[1959] 
del data_yb[1959]
del Amfsc_list[2230] 
del data_yb[2230]
del Amfsc_list[2610] 
del data_yb[2610]
del Amfsc_list[2888] 
del data_yb[2888]
del Amfsc_list[3212] 
del data_yb[3212]
del Amfsc_list[3470] 
del data_yb[3470]
del Amfsc_list[3635] 
del data_yb[3635]
del Amfsc_list[3831] 
del data_yb[3831]
del Amfsc_list[4059] 
del data_yb[4059]
del Amfsc_list[4295] 
del data_yb[4295]
del Amfsc_list[4657] 
del data_yb[4657]
del Amfsc_list[4933] 
del data_yb[4933]
del Amfsc_list[5214] 
del data_yb[5214]
del Amfsc_list[5495] 
del data_yb[5495]
del Amfsc_list[5805] 
del data_yb[5805]

modded_mfsc_concatenate = np.concatenate(Amfsc_list)
modded_mfsc_dstack = np.dstack(Amfsc_list)
modded_mfsc_hstack = np.hstack(Amfsc_list)
modded_mfsc_vstack = np.vstack(Amfsc_list)
modded_mfsc_stack = np.stack(Amfsc_list)

mfsc_input = [[]]
k = 0
i = 0