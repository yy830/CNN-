import librosa
import numpy as np

file_path = "C:/Users/daiyan/Desktop/Stu/DSP_design/free-spoken-digit-dataset-master/free-spoken-digit-dataset-master/recordings/0_george_2.wav"

def load_audio(file_path, sr=16000):
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr

def extract_mfcc(y, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def extract_spectrogram(y, sr):
    spect = librosa.stft(y)
    spect_db = librosa.amplitude_to_db(np.abs(spect))
    return spect_db
