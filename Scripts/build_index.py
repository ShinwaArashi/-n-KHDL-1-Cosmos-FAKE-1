import faiss
import numpy as np

# Load vectors đã lưu
vectors = np.load("embeddings/audio_vectors.npy").astype('float32')
filenames = np.load("embeddings/filenames.npy")

# Xây dựng FAISS index
index = faiss.IndexFlatL2(vectors.shape[1])  # Sử dụng khoảng cách Euclidean
index.add(vectors)

# Lưu index ra file
faiss.write_index(index, "embeddings/audio_index.faiss")
np.save("embeddings/filenames.npy", filenames)

print("✅ FAISS index created and saved.")
