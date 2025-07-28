import os
import numpy as np
import librosa
import faiss

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
    return features.astype("float32")

def get_metadata(file_path):
    filename = os.path.basename(file_path)
    try:
        name, bpm, key, type_ = filename.replace(".wav", "").split("_")
        return {
            "path": file_path,
            "filename": name,
            "bpm": int(bpm),
            "key": key,
            "type": type_
        }
    except Exception as e:
        print(f"⚠️ Lỗi khi parse metadata từ {filename}: {e}")
        return None

def scan_and_index_folder(folder_path):
    vectors = []
    filenames = []

    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(".wav"):
                path = os.path.join(root, f)
                try:
                    feat = extract_features(path)
                    vectors.append(feat)
                    filenames.append(path)
                except Exception as e:
                    print(f"❌ Lỗi file {path}: {e}")

    if not vectors:
        raise ValueError("⚠️ Không tìm thấy file WAV nào hợp lệ trong thư mục.")

    dim = vectors[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))

    sample_metadata = []
    for path in filenames:
        meta = get_metadata(path)
        if meta:
            sample_metadata.append(meta)

    return index, sample_metadata
