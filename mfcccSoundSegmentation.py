# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 05:02:51 2018

@author: Hrishikesh S
"""
# cd "Desktop/SECOND YEAR/RESEARCH"

import librosa
import numpy as np
import pandas as pd
from librosa import display
import matplotlib.pyplot as plt
import math
import tensorflow.python.keras.callbacks
import keras.callbacks
import time
import tensorboard
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import InputLayer, Conv1D, Dense, Flatten, MaxPool1D
from tensorflow.python.keras.models import Sequential

"""For using TensorBoard"""
def modelTrainingTensorBoard(model, feature_x, labels_y):
    model.fit(feature_x, labels_y, batch_size=20, epochs=5, 
              callbacks = [tensorflow.keras.callbacks.TensorBoard(log_dir="logs/final/{}".format(time()), 
                                                       histogram_freq=1, write_graph=True, write_images=True)])
    return model

"""For MFCC extraction"""
def mfcc_extraction(folder_name, tmp):
    data_y = []
    mfccs_list = []
    for j in range(tmp.shape[0]):
        for i in range(1, tmp.shape[1] - 1):
            try: 
                data, sampling_rate = librosa.load(folder_name + tmp.iloc[j, 0].split('.')[0] +'.wav' )
                print(sampling_rate)
                """we are taking value at every 25ms, so to decide size of the array, from which we
                take the mfcc"""
                size_factor = 25*(sampling_rate//1000)
                #temp_data = data[int(tmp.iloc[j, i]):int(tmp.iloc[j, i+1])]
                temp_label = tmp.iloc[:, i].name.split('.')[0]
                #print(temp_data)
                print(temp_label)
                start = int(tmp.iloc[j, i])
                end = int(tmp.iloc[j, i+1])
                """extracting features"""
                while(start<=end):
                    temp_data = data[start:start+size_factor]
                    start = start + size_factor
                    mfccs = np.mean(librosa.feature.mfcc(y=temp_data, sr=sampling_rate, n_mfcc=13).T, axis = 0)
                    plt.title(temp_label)
                    display.waveplot(mfccs, sr=sampling_rate)
                    plt.savefig("Aplots/13MFCC/"+str(tmp.iloc[j, 0].split('.')[0])+"S"+str(start-size_factor)+"E"+str(start)+".png")
                    plt.close()
                    mfccs_list.append(mfccs)
                    data_y.append(temp_label)
                print("Status : Success!")
            except:
                print("Status : Failed!")
                pass
    return mfccs_list, data_y;

def mfcc_preprocessing(mfcc_list, data_y):
    """padding (just in case)"""
    mfcc_list = pad_sequences(mfcc_list, maxlen=13, dtype='float', padding='post', truncating='post', value=0.)
    """ normalizing data """
    mfcc_list = mfcc_list / np.max(mfcc_list)
    """ creating appropriate data for my analysis """
    mfcc_list = mfcc_list[:,:,np.newaxis]
    """assigning integral labels"""
    data_y = pd.Series(data_y)
    data_y.value_counts()    
    data_y = data_y.map({'S1':0, 'S2':1}).values
    return mfcc_list, data_y;

def createMFCCModel(model, mfcc_list):
    print(mfcc_list.shape[1:])
    model.add(InputLayer(input_shape=mfcc_list.shape[1:]))
    """Hidden Layer 1"""
    model.add(Conv1D(filters=3, kernel_size=3, activation='relu'))
    model.add(MaxPool1D(strides=4))
    """Hidden Layer 2"""
    model.add(Conv1D(filters=2, kernel_size=2, activation='relu'))
    model.add(MaxPool1D(strides=2))
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

def modelMFCCTesting(model, folder_name, filename, flag):
    """this is how our test array should be processed"""
    found = False
    csv_name = ""
    if(folder_name == "Atraining_normal"):
        csv_name = "Atraining_normal_seg.csv"
    elif(folder_name == "Training B Normal"):
        csv_name = "Btraining_normal_seg.csv"
    else:
        print("Error")
        return;
    tmp = pd.read_csv(csv_name)
    tmp.head()
    print(tmp.shape[0],tmp.shape[1])
    test_labels = []
    test_data = []
    for j in range(tmp.shape[0]):
        for i in range(1, tmp.shape[1] - 1):
            try: 
                fname = folder_name + "/" + tmp.iloc[j, 0].split('.')[0] 
                #print(fname)
                #print(folder_name +"/"+ filename)
                if(fname == folder_name +"/"+ filename):
                    print(fname)
                    print(folder_name +"/"+ filename)
                    found = True
                    data, sampling_rate = librosa.load(fname + '.wav')
                    print(sampling_rate)
                    """we are taking value at every 25ms, so to decide size of the array, from which we
                    take the mfcc"""
                    size_factor = 25*(sampling_rate//1000)
                    #temp_data = data[int(tmp.iloc[j, i]):int(tmp.iloc[j, i+1])]
                    temp_label = tmp.iloc[:, i].name.split('.')[0]
                    #print(temp_data)
                    print(temp_label)
                    start = int(tmp.iloc[j, i])
                    end = int(tmp.iloc[j, i+1])
                    """extracting features"""
                    while(start<=end):
                        temp_data = data[start:start+size_factor]
                        start = start + size_factor
                        mfccs = np.mean(librosa.feature.mfcc(y=temp_data, sr=sampling_rate, n_mfcc=13).T, axis = 0)
                        #plt.title(temp_label)
                        #display.waveplot(mfccs, sr=sampling_rate)
                        #plt.savefig("Aplots/13MFCC/"+str(tmp.iloc[j, 0].split('.')[0])+"S"+str(start-size_factor)+"E"+str(start)+".png")
                        #plt.close()
                        test_data.append(mfccs)
                        test_labels.append(temp_label)
                        print("Status : Success!")
            except:
                print("Status : Failed!")
                pass
    if(found):
        print(test_data)
        """padding (just in case)"""
        test_data = pad_sequences(test_data, maxlen=13, dtype='float', padding='post', truncating='post', value=0.)
        """ normalizing data """
        test_data = test_data / np.max(test_data)
        """ creating appropriate data for my analysis"""
        test_data = test_data[:,:,np.newaxis]
        """converting labels into numeric data"""
        test_labels = pd.Series(test_labels)
        test_labels.value_counts()    
        test_labels = test_labels.map({'S1':0, 'S2':1}).values
        result = model.predict(test_data)
        print(result)
        prediction  = []
        for i in range((result.shape)[0]):
            if result[i][0] > 0.5:
                prediction.append(1)
            if result[i][0] <= 0.5:
                prediction.append(0)
        print(prediction)
        if(flag == 0):
            """To see how accurate the model is"""
            accuracy_f = (prediction == test_labels).mean()
            print(accuracy_f)
    else:
        print("File not found!")

"""Training with dataset A"""
temp = pd.read_csv('Atraining_normal_seg.csv')
temp.head()
print(temp.shape[0],temp.shape[1])

mfcc_list, data_y = mfcc_extraction("Atraining_normal/", temp)
mfcc_list, data_y = mfcc_preprocessing(mfcc_list, data_y)

"""Training and testing with 50 mfcc co-efficients"""
model_mfcc = Sequential()
model_mfcc = createMFCCModel(model_mfcc, mfcc_list)
model_mfcc = modelMFCCTraining(model_mfcc, mfcc_list, data_y)
# model_amp = modelTrainingTensorBoard(model_amp, data_x, data_y)

"""extracting for 13 mfcc co-efficients"""
mfcc_13, data_y = mfcc_extraction("Atraining_normal/", temp)
mfcc_13, data_y = mfcc_preprocessing(mfcc_13, data_y)

model_mfcc_13 = Sequential()
model_mfcc_13 = createMFCCModel(model_mfcc_13, mfcc_13)
model_mfcc_13 = modelMFCCTraining(model_mfcc_13, mfcc_13, data_y)

"""Trying with test data"""
modelMFCCTesting(model_mfcc_13, "Training B Normal", "103_1305031931979_D3", 0)