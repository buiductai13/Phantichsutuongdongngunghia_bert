import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.decomposition import PCA
import seaborn as sns

# Hàm load model và tokenizer (giả sử mô hình tương đồng từ trước)
@st.cache_resource
def load_model_and_tokenizer():
    model_path = "model/modelphobert_similarity_model_2"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_model_and_tokenizer()
# hàm tính ddienmr tương đồng  
def predict_similarity(sentence1, sentence2):
    """
    Dự đoán điểm tương đồng giữa hai câu.
    Giả sử mô hình trả về giá trị từ 0 đến 5.
    """
    inputs = tokenizer(sentence1, sentence2, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits.item()
    return score

def compute_similarity(sentence1, sentence2):
    """
    Hàm compute_similarity:
    - Gọi predict_similarity để tính điểm tương đồng.
    - Giả sử giá trị trả về đã nằm trong khoảng [0,5].
    """
    return predict_similarity(sentence1, sentence2)

def get_embedding(sentence):
    """
    Hàm get_embedding:
    - Lấy vector nhúng từ mô hình (dùng CLS embedding).
    """
    backbone = model.base_model
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    inputs = {k: v.to(device) for k,v in inputs.items()}
    backbone.eval()
    with torch.no_grad():
        outputs = backbone(**inputs)
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    return cls_embedding
#hàm tóm tắt văn bản nội dung điêm tương đồng
def advanced_extractive_summary_short(text, max_sentences=4):
    """
    Tóm tắt văn bản ngắn gọn sử dụng mô hình tương đồng ngữ nghĩa:
    - Phân đoạn văn bản thành các câu.
    - Tính điểm trung bình độ tương đồng của mỗi câu với các câu khác.
    - Chọn các câu đại diện có điểm số cao nhất (ưu tiên nội dung chính).
    - Trả về tóm tắt ngắn gọn với số câu giới hạn bởi max_sentences.
    """
    # Phân đoạn văn bản thành các câu
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) == 0:
        return "Văn bản quá ngắn để tóm tắt."
    if len(sentences) == 1:
        return sentences[0]

    # Tính ma trận tương đồng
    n = len(sentences)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:  # Không cần tính điểm tương đồng giữa câu với chính nó
                sim_matrix[i, j] = compute_similarity(sentences[i], sentences[j])

    # Tính điểm trung bình cho mỗi câu (đại diện mức độ quan trọng)
    avg_sim = sim_matrix.mean(axis=1)

    # Sắp xếp các câu theo điểm số trung bình giảm dần
    sorted_indices = np.argsort(avg_sim)[::-1]

    # Chọn số câu theo yêu cầu (max_sentences)
    selected_sentences = [sentences[idx] for idx in sorted_indices[:max_sentences]]

    # Ghép các câu được chọn thành một đoạn tóm tắt ngắn gọn
    summary = " ".join(selected_sentences)
    return summary



# Tiêu đề cho ứng dụng
st.title("Phân Tích Tương Đồng Ngữ Nghĩa Mô Hình BERT")

# Tab điều hướng
tab1, tab2, tab3 = st.tabs(["Nhập từng câu", "Tóm tắt văn bản", "So sánh đoạn văn để phát hiện sao chép "])

# Tab 1: Nhập từng câu
with tab1:
    num_sentences = st.number_input("Chọn số lượng câu cần so sánh", min_value=2, max_value=10, step=1, key="num_sentences")

    sentences = []
    for i in range(num_sentences):
        sentence = st.text_input(f"Nhập câu {i + 1}", key=f"sentence_{i}")
        if sentence:
            sentences.append(sentence)

    if st.button("So sánh độ tương đồng", key="button_single"):
        if len(sentences) == num_sentences:
            similarities = np.zeros((num_sentences, num_sentences))
            
            for i in range(num_sentences):
                for j in range(i, num_sentences):
                    if i != j:
                        similarity = compute_similarity(sentences[i], sentences[j])
                        similarities[i][j] = similarity
                        similarities[j][i] = similarity
                    else:
                        similarities[i][j] = 5.0
                        
            st.subheader("Điểm Tương Đồng")
            overall_similarity = np.mean(similarities)
            for i in range(num_sentences):
                for j in range(i + 1, num_sentences):
                    st.write(f"Độ tương đồng giữa câu {i + 1} và câu {j + 1}: {similarities[i][j]:.2f}/5")
# Tab 2: Tóm tắt văn bản
with tab2:
    st.subheader("Nhập nội dung văn bản để tóm tắt")
    text_to_summarize = st.text_area("Nhập văn bản", height=150, key="text_to_summarize")

    if st.button("Tóm tắt", key="button_summary"):
        if text_to_summarize.strip():
            summary = advanced_extractive_summary_short(text_to_summarize, max_sentences=4)  
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Nội dung gốc:")
                st.write(text_to_summarize) 
            with col2:
                st.subheader("Tóm tắt kết quả:")
                st.write(summary) 
        else:
            st.warning("Vui lòng nhập văn bản để tóm tắt.")
            
# Tab 3: Kiểm tra đạo văn
with tab3:
    st.subheader("Nhập hai đoạn văn để so sanh tương đồng ")
    text_to_check = st.text_area("Nhập đoạn văn", height=150, key="text_to_check")
    reference_text = st.text_area("Nhập đoạn văn tham chiếu để so sánh", height=150, key="reference_text")

    if st.button("Kiểm tra đạo văn", key="button_plagiarism"):
        if text_to_check and reference_text:
            similarity = compute_similarity(text_to_check, reference_text)
            st.write(f"Độ tương đồng giữa đoạn văn gốc và đoạn văn tham chiếu: {similarity:.2f}/5")
        else:
            st.warning("Vui lòng nhập cả đoạn văn và đoạn văn tham chiếu.")

