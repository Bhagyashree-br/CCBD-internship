# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:24:37 2018

@author: Hrishikesh S.
         Kartik A.N.
         Bhagyashree B.R.
"""

# cd "Desktop/SECOND YEAR/RESEARCH"

import librosa
import numpy as np
import pandas as pd
from librosa import display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

n_hidden = 0
"""Method from Urban Sound Classification"""
def RNN(x, weight, bias):
    cell = rnn_cell.LSTMCell(n_hidden,state_is_tuple = True)
    cell = rnn_cell.MultiRNNCell([cell] * 2)
    output, state = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)
    output = tf.transpose(output, [1, 0, 2])
    last = tf.gather(output, int(output.get_shape()[0]) - 1)
    return tf.nn.softmax(tf.matmul(last, weight) + bias)

prediction = RNN(x, weight, bias)

"""Method 2"""
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking, Bidirectional, InputLayer
from keras.optimizers import Adam
from keras.utils import plot_model

def createRNNModel(model, data):
    model.add(InputLayer(input_shape=data.shape[1:]))
    model.add(LSTM(units=64, dropout=0.1, recurrent_dropout=0.1, activation='tanh'))
    model.add(Dense(units=128, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','mae'])
    """ model architecture """
    model.summary()
    return model