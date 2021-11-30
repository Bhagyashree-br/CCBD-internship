# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 06:57:31 2018

Segmentation of S1 and S2 phonographic sounds

@author: a. Hrishikesh S.
         b. Karthik A.N.
         c. Bhagyashree B.R.
Work to be done ::::
        1. Getting the test array properly using loops, 
           dealing with numpy.ndarray  ---> Karthik A.N. (Done)
        2. Train with dataset B and also more features
           (mffc, chroma, melspectrograms) ---> Hrishikesh S. (Done)
        3. Deal with noises, extra heart sounds and murmurs 
           (probably an unsupervised approach for that) ---> ???
        4. Location of S1 and S2 sounds occurrences :::
            Hint : Segment audio samples ---> Bhagyashree B.R.
        5. Train new model only on Atrial Fibrillation dataset ---> ???
        6. Create images for every feature ---> ???
"""
# cd "Desktop/SECOND YEAR/RESEARCH"

import librosa
import numpy as np
import pandas as pd
from librosa import display
import matplotlib.pyplot as plt
import math

#data, sampling_rate = librosa.load('Atraining_normal/201102081321.wav', sr=44100 )
#display.waveplot(data, sr=sampling_rate)


""" extracting raw data from the files """
"""features that we are currently working on 
        1. Amplitude
        2. MFCC
        3. Chroma
        4. Melspectrogram
        5. Tonnetz
        6. Spectral contrast
"""
def data_extractor(folder_name, tmp):
    data_x = []
    data_y = []
    mfccs_list = []
    chroma_list = []
    mel_list = []
    contrast_list = []
    tonnetz_list = []
    for j in range(tmp.shape[0]):
        for i in range(1, tmp.shape[1] - 1):
            try: 
                data, sampling_rate = librosa.load(folder_name + tmp.iloc[j, 0].split('.')[0] +'.wav' )
                print(sampling_rate)
                temp_data = data[int(tmp.iloc[j, i]):int(tmp.iloc[j, i+1])]
                temp_label = tmp.iloc[:, i].name.split('.')[0]
                print(temp_data)
                print(temp_label)
                """extracting features"""
                stft = np.abs(librosa.stft(temp_data))
                mfccs = np.mean(librosa.feature.mfcc(y=temp_data, sr=sampling_rate, n_mfcc=50).T, axis = 0)
                chroma = np.mean(librosa.feature.chroma_stft(S = stft, sr = sampling_rate).T, axis = 0)
                mel = np.mean(librosa.feature.melspectrogram(temp_data, sr = sampling_rate).T, axis = 0)
                contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sampling_rate).T, axis=0)
                tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(temp_data), sr=sampling_rate).T, axis =0)
                """plotting all features"""
                #plt.title(temp_label)
                #display.waveplot(temp_data, sr=sampling_rate)
                #plt.savefig("Bplots/Amplitude/"+str(tmp.iloc[j, 0].split('.')[0])+"S"+str(tmp.iloc[j, i])+"E"+str(tmp.iloc[j, i+1])+".png")
                #plt.close()
                #plt.title(temp_label)
                #display.waveplot(mfccs, sr=sampling_rate)
                #plt.savefig("Bplots/MFCC/"+str(tmp.iloc[j, 0].split('.')[0])+"S"+str(tmp.iloc[j, i])+"E"+str(tmp.iloc[j, i+1])+".png")
                #plt.close()
                #plt.title(temp_label)
                #display.waveplot(chroma, sr=sampling_rate)
                #plt.savefig("Bplots/Chroma/"+str(tmp.iloc[j, 0].split('.')[0])+"S"+str(tmp.iloc[j, i])+"E"+str(tmp.iloc[j, i+1])+".png")
                #plt.close()
                #plt.title(temp_label)
                #display.waveplot(mel, sr=sampling_rate)
                #plt.savefig("Bplots/MEL/"+str(tmp.iloc[j, 0].split('.')[0])+"S"+str(tmp.iloc[j, i])+"E"+str(tmp.iloc[j, i+1])+".png")
                #plt.close()
                #plt.title(temp_label)
                #display.waveplot(contrast, sr=sampling_rate)
                #plt.savefig("Bplots/Contrast/"+str(tmp.iloc[j, 0].split('.')[0])+"S"+str(tmp.iloc[j, i])+"E"+str(tmp.iloc[j, i+1])+".png")
                #plt.close()
                #plt.title(temp_label)
                #display.waveplot(tonnetz, sr=sampling_rate)
                #plt.savefig("Bplots/Tonnetz/"+str(tmp.iloc[j, 0].split('.')[0])+"S"+str(tmp.iloc[j, i])+"E"+str(tmp.iloc[j, i+1])+".png")
                #plt.close()
                """appending to list"""
                data_x.append(temp_data)
                mfccs_list.append(mfccs)
                data_y.append(temp_label)
                chroma_list.append(chroma)
                mel_list.append(mel)
                contrast_list.append(contrast)
                tonnetz_list.append(tonnetz)
            except:
                pass
    return data_x, data_y, mfccs_list, chroma_list, mel_list, contrast_list, tonnetz_list;

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

def data_preprocessing(data_x, data_y, mfcc_list, chroma_list, mel_list, contrast_list, tonnetz_list):
    """ giving same shape to all the extracted data """
    data_x = pad_sequences(data_x, maxlen=20000, dtype='float', padding='post', truncating='post', value=0)
    mfcc_list = pad_sequences(mfcc_list, maxlen=20000, dtype='float', padding='post', truncating='post', value=0.)
    chroma_list = pad_sequences(chroma_list, maxlen=20000, dtype='float', padding='post', truncating='post', value=0.)
    mel_list = pad_sequences(mel_list, maxlen=20000, dtype='float', padding='post', truncating='post', value=0.)
    contrast_list = pad_sequences(contrast_list, maxlen=20000, dtype='float', padding='post', truncating='post', value=0.)
    tonnetz_list = pad_sequences(tonnetz_list, maxlen=20000, dtype='float', padding='post', truncating='post', value=0.)
    """ normalizing data """
    data_x = data_x / np.max(data_x)
    mfcc_list = mfcc_list / np.max(mfcc_list)
    chroma_list = chroma_list / np.max(chroma_list)
    mel_list = mel_list / np.max(mel_list)
    contrast_list = contrast_list / np.max(contrast_list)
    tonnetz_list = tonnetz_list / np.max(tonnetz_list)
    """ creating appropriate data for my analysis """
    data_x = data_x[:,:,np.newaxis]
    mfcc_list = mfcc_list[:,:,np.newaxis]
    chroma_list = chroma_list[:,:,np.newaxis]
    mel_list = mel_list[:,:,np.newaxis]
    contrast_list = contrast_list[:,:,np.newaxis]
    tonnetz_list = tonnetz_list[:,:,np.newaxis]
    """assigning integral labels"""
    data_y = pd.Series(data_y)
    data_y.value_counts()    
    data_y = data_y.map({'S1':0, 'S2':1}).values
    return data_x, data_y, mfcc_list, chroma_list, mel_list, contrast_list, tonnetz_list;

"""load the file and return the sliced & re-appended numpy array"""
def loadFile(filename):
    data, sr = librosa.load(filename)
    print(len(data))
    a = np.array(test_data[0:20000])
    it = len(data)//20000
    x=0
    for i in range(it):
        l=np.array(test_data[x:(x+20000)])
        a=np.vstack((a,l))
        x+=20000
    """data-preprocessing (padding{ not need for this most probably }, normalization, reshape)"""
    """ giving same shape to all the extracted data """
    #a = pad_sequences(a, maxlen=20000, dtype='float', padding='post', truncating='post', value=0)
    """ normalizing data """
    a = a / np.max(a)
    """ creating appropriate data for my analysis """
    a = a[:,:,np.newaxis]
    return a;

""" building CNN """
from tensorflow.python.keras.layers import InputLayer, Conv1D, Dense, Flatten, MaxPool1D
from tensorflow.python.keras.models import Sequential

def createModel(model, mfcc_list):
    print(mfcc_list.shape[1:])
    model.add(InputLayer(input_shape=mfcc_list.shape[1:]))
    """Hidden Layer 1"""
    model.add(Conv1D(filters=10, kernel_size=20, activation='tanh'))
    model.add(MaxPool1D(strides=3))
    """Hidden Layer 2"""
    model.add(Conv1D(filters=10, kernel_size=20, activation='tanh'))
    model.add(MaxPool1D(strides=3))
    """Hidden Layer 3"""
    model.add(Conv1D(filters=10, kernel_size=20, activation='tanh'))
    model.add(MaxPool1D(strides=3))
    """Flattening and Output"""
    model.add(Flatten())
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','mae'])
    """ model architecture """
    model.summary()
    return model

"""training model"""
def modelTraining(model, feature_x, labels_y):
    model.fit(feature_x, labels_y, batch_size=20, epochs=5)
    return model;

def modelTesting(model, test_data, data_y, flag):
    result = model.predict(test_data)
    #print(result)
    prediction  = []
    for i in range((result.shape)[0]):
        if result[i][0] > 0.5:
            prediction.append(1)
        if result[i][0] <= 0.5:
            prediction.append(0)
    print(prediction)
    if(flag == 0):
        """To see how accurate the model is"""
        accuracy_f = (prediction == data_y).mean()
        print(accuracy_f)


"""Training with dataset A"""
temp = pd.read_csv('Atraining_normal_seg.csv')
temp.head()
print(temp.shape[0],temp.shape[1])

data_x, data_y, mfcc_list, chroma_list, mel_list, contrast_list, tonnetz_list = data_extractor("Atraining_normal/", temp)

data_x, data_y, mfcc_list, chroma_list, mel_list, contrast_list, tonnetz_list = data_preprocessing(data_x, data_y, mfcc_list, chroma_list, mel_list, contrast_list, tonnetz_list)

"""Training and testing with amplitudes"""
model_amp = Sequential()
model_amp = createModel(model_amp, mfcc_list)
model_amp = modelTraining(model_amp, data_x, data_y)
# model_amp = modelTrainingTensorBoard(model_amp, data_x, data_y)
modelTesting(model_amp, data_x, data_y, 0)

"""Trying with test data"""
"""librosa load each of them and append them onto a list"""
test_data, test_sr = librosa.load("Atraining_normal/201101070538.wav")
"""this is how our test array should be processed"""
a = np.array([test_data[0:10000], test_data[10000:20000], test_data[20000:30000],
              test_data[30000:40000], test_data[40000:50000], test_data[50000:60000],
              test_data[60000:70000], test_data[70000:80000], test_data[80000:90000],
              test_data[90000:100000], test_data[10000:120000]])
print(a.shape)

b = loadFile("Atraining_normal/201101070538.wav")
print(b.shape)
"""making new predictions from newly loaded data""" 
modelTesting(model_amp, b, [], 1)

"""Training with dataset B"""
temp = pd.read_csv('Btraining_normal_seg.csv')
temp.head()
print(temp.shape[0],temp.shape[1])

data_x, data_y, mfcc_list, chroma_list, mel_list, contrast_list, tonnetz_list = data_extractor("Training B Normal/", temp)

data_x, data_y, mfcc_list, chroma_list, mel_list, contrast_list, tonnetz_list = data_preprocessing(data_x, data_y, mfcc_list, chroma_list, mel_list, contrast_list, tonnetz_list)

"""Creating, training and testing model with amplitudes"""
model = Sequential()
model = createModel(model, mfcc_list)
model = modelTraining(model, data_x, data_y)    #around 5 epochs
modelTesting(model, data_x, data_y, 0)
modelTesting(model, a, [] ,flag = 1)

"""Creating, training and testing model with mfccs"""
model_mfcc = Sequential()
model_mfcc = createModel(model_mfcc, mfcc_list)
model_mfcc = modelTraining(model_mfcc, mfcc_list, data_y)   #around 5 epochs
modelTesting(model_mfcc, mfcc_list, data_y, 0)
modelTesting(model_mfcc, a, [] ,flag = 1)

"""Creating, training and testing model with melspectrograms"""
model_mel = Sequential()
model_mel = createModel(model_mel, mel_list)
model_mel = modelTraining(model_mel, mel_list, data_y)   #around 5 epochs
modelTesting(model_mel, mel_list, data_y, 0)
modelTesting(model_mel, a, [] ,flag = 1)

"""Creating, training and testing model with tonnetz"""
model_tonnetz = Sequential()
model_tonnetz = createModel(model_tonnetz, tonnetz_list)
model_tonnetz = modelTraining(model_tonnetz, tonnetz_list, data_y)   #around 5 epochs
modelTesting(model_tonnetz, tonnetz_list, data_y, 0)
modelTesting(model_tonnetz, a, [] ,flag = 1)

"""Creating, training and testing model with contrast"""
model_contrast = Sequential()
model_contrast = createModel(model_contrast, contrast_list)
model_contrast = modelTraining(model_contrast, contrast_list, data_y)   #around 5 epochs
modelTesting(model_contrast, contrast_list, data_y, 0)
modelTesting(model_contrast, a, [] ,flag = 1)

"""Creating, training and testing model with contrast"""
model_chroma = Sequential()
model_chroma = createModel(model_chroma, chroma_list)
model_chroma = modelTraining(model_chroma, chroma_list, data_y)   #around 5 epochs
modelTesting(model_chroma, chroma_list, data_y, 0)
modelTesting(model_chroma, a, [] ,flag = 1)