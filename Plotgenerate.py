"""
Use command prompt or terminal for this
"""
# cd "Desktop/SECOND YEAR/RESEARCH"
import librosa
import numpy as np
import pandas as pd
from librosa import display
import matplotlib.pyplot as plt

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
                #temp_data = data[int(tmp.iloc[j, i]):int(tmp.iloc[j, i+1])]
                beat_loc = int(tmp.iloc[j, i])
                temp_data = data[beat_loc - 1000 : beat_loc + 1000]
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
                plt.title(temp_label)
                display.waveplot(temp_data, sr=sampling_rate)
                plt.savefig("Newplots/Amplitude/"+str(tmp.iloc[j, 0].split('.')[0])+"S"+str(beat_loc - 1000)+"E"+str(beat_loc + 1000)+".png")
                plt.close()
                plt.title(temp_label)
                display.waveplot(mfccs, sr=sampling_rate)
                plt.savefig("Newplots/MFCC/"+str(tmp.iloc[j, 0].split('.')[0])+"S"+str(beat_loc - 1000)+"E"+str(beat_loc + 1000)+".png")
                plt.close()
                plt.title(temp_label)
                display.waveplot(chroma, sr=sampling_rate)
                plt.savefig("Newplots/Chroma/"+str(tmp.iloc[j, 0].split('.')[0])+"S"+str(beat_loc - 1000)+"E"+str(beat_loc + 1000)+".png")
                plt.close()
                plt.title(temp_label)
                display.waveplot(mel, sr=sampling_rate)
                plt.savefig("Newplots/MEL/"+str(tmp.iloc[j, 0].split('.')[0])+"S"+str(beat_loc - 1000)+"E"+str(beat_loc + 1000)+".png")
                plt.close()
                plt.title(temp_label)
                display.waveplot(contrast, sr=sampling_rate)
                plt.savefig("Newplots/Contrast/"+str(tmp.iloc[j, 0].split('.')[0])+"S"+str(beat_loc - 1000)+"E"+str(beat_loc + 1000)+".png")
                plt.close()
                plt.title(temp_label)
                display.waveplot(tonnetz, sr=sampling_rate)
                plt.savefig("Newplots/Tonnetz/"+str(tmp.iloc[j, 0].split('.')[0])+"S"+str(beat_loc - 1000)+"E"+str(beat_loc + 1000)+".png")
                plt.close()

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

temp = pd.read_csv('Atraining_normal_seg.csv')
temp.head()
print(temp.shape[0],temp.shape[1])

data_x, data_y, mfcc_list, chroma_list, mel_list, contrast_list, tonnetz_list = data_extractor("Atraining_normal/", temp)
