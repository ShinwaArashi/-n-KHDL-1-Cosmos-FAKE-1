import librosa
import numpy as np
import os
import soundfile as sf

def extract_features(file_path):
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

    return features

def process_directory(input_dir, output_file):
    features_list = []
    filenames = []

    for file in os.listdir(input_dir):
        if file.endswith(".wav"):
            path = os.path.join(input_dir, file)
            features = extract_features(path)
            features_list.append(features)
            filenames.append(file)

    np.save("embeddings/audio_vectors.npy", np.array(features_list))
    np.save("embeddings/filenames.npy", np.array(filenames))
    print("✓ Trích xuất xong và lưu vector thành công.")

if __name__ == "__main__":
    process_directory("data/raw", "embeddings/audio_vectors.npy")
