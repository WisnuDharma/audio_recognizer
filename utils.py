# utils.py
# Tempat fungsi bantu untuk ekstraksi fitur audio, dsb. 

import numpy as np
import librosa

def extract_features_full(y_or_path, sr=22050, n_mfcc=40):
    # Jika input string, load file
    if isinstance(y_or_path, str):
        y, sr = librosa.load(y_or_path, sr=sr)
    else:
        y = y_or_path
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    # Chroma
    stft = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel, axis=1)
    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    # Gabungkan semua fitur
    features = np.hstack([
        mfcc_mean,
        chroma_mean,
        mel_mean,
        contrast_mean,
        zcr_mean
    ])
    return features 