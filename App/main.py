import os
import sys
import gradio as gr
import numpy as np
import librosa

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from Scripts.search_engine import scan_and_index_folder, search_dynamic
from Embeddings.embeddings import extract_embedding

# Global FAISS index + file path list
faiss_index = None
sample_db = []

def scan_folder(folder_path):
    global faiss_index, sample_db
    faiss_index, sample_db = scan_and_index_folder(folder_path)

    if not os.path.exists(folder_path):
        return "❌ Thư mục không tồn tại!", None

    try:
        print(f"[INFO] Scanning folder: {folder_path}")
        faiss_index, file_paths = scan_and_index_folder(folder_path)
        return f"✅ Đã index {len(file_paths)} file sample.", None
    except Exception as e:
        return f"❌ Lỗi khi scan thư mục: {str(e)}", None

def search_similar(uploaded_audio):
    global faiss_index, file_paths

    if uploaded_audio is None:
        return "⚠️ Bạn chưa upload file.", []

    if faiss_index is None or not file_paths:
        return "⚠️ Bạn chưa scan thư mục nào.", []

    try:
        y, sr = librosa.load(uploaded_audio, sr=None)
        emb = extract_embedding(y, sr)
        results = search_dynamic(emb, faiss_index, file_paths, top_k=5)

        display_results = []
        for path, score in results:
            meta = next((m for m in sample_db if m["path"] == path), None)
            if meta:
                label = f"{meta['filename']} | BPM: {meta['bpm']} | Key: {meta['key']} | {meta['type']} | Score: {score:.2f}"
                display_results.append((path, label))


        return "🎯 Kết quả giống nhất (Top 5):", display_results

    except Exception as e:
        return f"❌ Lỗi khi tìm kiếm: {str(e)}", []

with gr.Blocks(title="Fake COSMOS - Sample Finder") as demo:
    gr.Markdown("## 🔍 FAKE COSMOS - Sample Finder 🎶")
    
    with gr.Row():
        filter_key = gr.Textbox(label="🎼 Key (ví dụ: C, D#, F...)")
        filter_bpm = gr.Slider(minimum=60, maximum=200, label="🕒 BPM", step=1)
        filter_type = gr.Dropdown(choices=["oneshot", "loop", "all"], value="all", label="🔁 Type")
        name_query = gr.Textbox(label="🔤 Tìm theo tên")
        

    with gr.Row():
        folder_input = gr.Textbox(label="📂 Đường dẫn thư mục chứa WAV")
        scan_btn = gr.Button("📁 Quét thư mục")
    scan_output = gr.Textbox(label="📝 Trạng thái")

    with gr.Row():
        upload_input = gr.Audio(label="🎧 Upload file WAV để tìm")
        search_btn = gr.Button("🔍 Tìm file tương tự")

    result_text = gr.Textbox(label="📋 Thông báo")
    result_gallery = gr.Gallery(label="🎼 Kết quả giống nhất", columns=1)

    scan_btn.click(scan_folder, inputs=folder_input, outputs=[scan_output])
    search_btn.click(search_similar, inputs=upload_input, outputs=[result_text, result_gallery])

# Nút quét thư mục
scan_btn.click(scan_folder, inputs=folder_input, outputs=[scan_output])

# Nút tìm kiếm file tương tự
search_btn.click(
    search_similar, 
    inputs=[upload_input, filter_key, filter_bpm, filter_type, name_query],
    outputs=[result_text, result_gallery]
)

if __name__ == "__main__":
    demo.launch()
