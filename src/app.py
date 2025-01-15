import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.decomposition import PCA

# === Load model và tokenizer ===
@st.cache_resource
def load_model_and_tokenizer():
    """
    Hàm tải mô hình và tokenizer từ thư mục đã lưu.
    """
    model_path = "model/modelphobert_similarity_model_2"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_model_and_tokenizer()

# === Hàm tính điểm tương đồng ===
def compute_similarity(sentence1, sentence2):
    """
    Dự đoán điểm tương đồng giữa hai câu dựa trên mô hình.
    """
    inputs = tokenizer(sentence1, sentence2, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits.item()
    return score

# === Hàm lấy vector nhúng ===
def get_embedding(sentence):
    """
    Trích xuất vector nhúng (embedding) từ mô hình.
    """
    backbone = model.base_model
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    backbone.eval()
    with torch.no_grad():
        outputs = backbone(**inputs)
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    return cls_embedding

# === Hàm tóm tắt văn bản ===
def advanced_extractive_summary_short(text, max_sentences=3):
    """
    Tóm tắt văn bản ngắn gọn dựa trên mức độ tương đồng.
    """
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) < 2:
        return "Văn bản quá ngắn để tóm tắt."

    n = len(sentences)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim_matrix[i, j] = compute_similarity(sentences[i], sentences[j])
            sim_matrix[j, i] = sim_matrix[i, j]  # Ma trận đối xứng

    avg_sim = sim_matrix.mean(axis=1)
    sorted_indices = np.argsort(avg_sim)[::-1]
    selected_sentences = [sentences[idx] for idx in sorted_indices[:max_sentences]]

    return " ".join(selected_sentences)

# === Ứng dụng Streamlit ===
st.title("Phân Tích Tương Đồng Ngữ Nghĩa Mô Hình BERT")

# Tab điều hướng
tab1, tab2, tab3 = st.tabs(["Nhập từng câu", "Tóm tắt văn bản", "So sánh đoạn văn"])

# === Tab 1: So sánh từng câu ===
with tab1:
    num_sentences = st.number_input("Số lượng câu cần so sánh", min_value=2, max_value=10, step=1)
    sentences = [st.text_input(f"Nhập câu {i + 1}", key=f"sentence_{i}") for i in range(num_sentences)]

    if st.button("So sánh độ tương đồng"):
        if all(sentences):
            similarities = np.zeros((num_sentences, num_sentences))
            for i in range(num_sentences):
                for j in range(i, num_sentences):
                    similarities[i][j] = compute_similarity(sentences[i], sentences[j]) if i != j else 5.0
                    similarities[j][i] = similarities[i][j]

            st.subheader("Điểm Tương Đồng")
            for i in range(num_sentences):
                for j in range(i + 1, num_sentences):
                    st.write(f"Độ tương đồng giữa câu {i + 1} và câu {j + 1}: {similarities[i][j]:.2f}/5")

# === Tab 2: Tóm tắt văn bản ===
with tab2:
    st.subheader("Nhập nội dung văn bản để tóm tắt")
    text_to_summarize = st.text_area("Nhập văn bản", height=150, key="text_to_summarize")

    if st.button("Tóm tắt", key="button_summary"):
        if text_to_summarize.strip():
            summary = advanced_extractive_summary_short(text_to_summarize, max_sentences=6)  # max_sentences=6: Lấy câu
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Nội dung gốc:")
                st.write(text_to_summarize) 
            with col2:
                st.subheader("Tóm tắt kết quả:")
                st.write(summary) 
        else:
            st.warning("Vui lòng nhập văn bản để tóm tắt.")


# === Tab 3: So sánh đoạn văn ===
with tab3:
    text_to_check = st.text_area("Nhập đoạn văn cần kiểm tra", height=150)
    reference_text = st.text_area("Nhập đoạn văn tham chiếu", height=150)

    if st.button("Kiểm tra đạo văn"):
        if text_to_check.strip() and reference_text.strip():
            similarity = compute_similarity(text_to_check, reference_text)
            st.write(f"Độ tương đồng: {similarity:.2f}/5")
        else:
            st.warning("Vui lòng nhập đầy đủ hai đoạn văn.")
