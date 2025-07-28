import librosa
import numpy as np
from scipy.signal import spectrogram

def extract_spectrogram_embedding(path, sr=22050, n_mels=128):
    y, sr = librosa.load(path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    # Flatten để làm vector
    embedding = S_dB.flatten()
    return embedding / np.linalg.norm(embedding)  # chuẩn hóa vector
