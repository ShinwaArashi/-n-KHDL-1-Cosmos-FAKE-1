import numpy as np
import librosa

def extract_embedding(file_path):
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(zcr, axis=1),
        np.mean(centroid, axis=1)
    ])

    return features.astype("float32").reshape(1, -1)
