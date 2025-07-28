import os
import numpy as np
from embedding_engine import extract_spectrogram_embedding

def scan_and_embed(folder_path):
    vectors = []
    file_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                full_path = os.path.join(root, file)
                try:
                    emb = extract_spectrogram_embedding(full_path)
                    vectors.append(emb)
                    file_paths.append(full_path)
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
    return np.array(vectors), file_paths
