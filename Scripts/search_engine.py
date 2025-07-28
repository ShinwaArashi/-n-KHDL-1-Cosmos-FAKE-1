import os
import numpy as np
import faiss
import librosa
from Embeddings.embeddings import extract_embedding

def scan_and_index_folder(folder_path):
    embeddings = []
    file_paths = []

    for file in os.listdir(folder_path):
        if file.lower().endswith('.wav'):
            full_path = os.path.join(folder_path, file)
            y, sr = librosa.load(full_path, sr=None)
            emb = extract_embedding(y, sr)
            embeddings.append(emb)
            file_paths.append(full_path)

    # Convert to numpy array
    embedding_matrix = np.stack(embeddings).astype(np.float32)

    # FAISS index
    dim = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embedding_matrix)

    return index, file_paths

def search_dynamic(query_embedding, index, file_paths, top_k=5):
    query_vector = np.expand_dims(query_embedding, axis=0).astype(np.float32)
    D, I = index.search(query_vector, top_k)
    results = []

    for i, dist in zip(I[0], D[0]):
        if 0 <= i < len(file_paths):
            results.append((file_paths[i], dist))
    return results

def get_metadata(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        key = librosa.key.key_to_note(librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1).argmax())

        return {
            "filename": os.path.basename(file_path),
            "path": file_path,
            "bpm": round(tempo),
            "key": key,
            "tags": [],
            "type": "loop" if duration > 2 else "oneshot"
        }
    except Exception as e:
        print(f"[ERROR] Metadata failed for {file_path}: {str(e)}")
        return None
