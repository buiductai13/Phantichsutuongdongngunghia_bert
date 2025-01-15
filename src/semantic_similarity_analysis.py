import streamlit as st
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# Tải mô hình và tokenizer đã huấn luyện từ thư mục địa phương
@st.cache_resource
def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    return tokenizer, model

# Tải mô hình đã huấn luyện 'phobert_similarity_model'
tokenizer, model = load_model("model/phobert_similarity_model")

# Tải mô hình đã huấn luyện cho tóm tắt văn bản
@st.cache_resource
def load_summary_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model

summary_model_path = "model/fine_tuned_similarity_model"
summary_tokenizer, summary_model = load_summary_model(summary_model_path)

# Hàm để tính embedding của câu
def get_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    norm_embedding = embedding / np.linalg.norm(embedding)  # Chuẩn hóa vector
    return norm_embedding

# Hàm để tính độ tương đồng ngữ nghĩa giữa hai câu và chuyển đổi sang thang điểm 0-5
def compute_similarity(sentence1, sentence2):
    embedding1 = get_embedding(sentence1)
    embedding2 = get_embedding(sentence2)
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]  # Công thức cho độ tương đồng cosine giữa hai vector
    return (similarity + 1) * 2.5  # Chuyển đổi sang thang điểm 0-5

# Hàm để tóm tắt văn bản
def extractive_summary(text, max_sentences=1):
    # Chia văn bản thành từng câu
    sentences = text.split(". ")
    
    # Kiểm tra nếu không có câu nào
    if len(sentences) == 0:
        return "Không có nội dung để tóm tắt."

    # Sử dụng câu đầu tiên làm câu tham chiếu
    reference_sentence = sentences[0]

    # Tính điểm tương đồng cho mỗi câu với câu tham chiếu
    sentence_scores = []
    for sentence in sentences:
        # Token hóa câu tham chiếu và câu hiện tại
        inputs = summary_tokenizer(
            reference_sentence, sentence,
            padding="max_length", truncation=True, max_length=128,
            return_tensors="pt"
        )

        # Tính điểm của câu
        with torch.no_grad():
            outputs = summary_model(**inputs)
            score = outputs.logits.squeeze().item()
            sentence_scores.append((sentence, score))

    # Sắp xếp các câu theo thứ tự điểm số giảm dần
    sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)

    # Chọn ra các câu có điểm cao nhất cho tóm tắt
    summary_sentences = [sent for sent, score in sorted_sentences[:max_sentences]]

    # Tạo tóm tắt từ các câu đã chọn
    summary = " ".join(summary_sentences).strip()  # Kết hợp các câu thành một chuỗi duy nhất
    return summary if summary else "Không có câu nào để tóm tắt."


# Hàm để tạo biểu đồ scatter với thanh màu
def create_scatter_plot(df, num_components, avg_similarities):
    if num_components == 3:
        fig = px.scatter_3d(df, x='x', y='y', z='z', hover_name='sent', 
                            color=avg_similarities, color_continuous_scale=px.colors.sequential.Plasma,
                            labels={'color': 'Độ tương đồng trung bình'})
    else:
        fig = px.scatter(df, x='x', y='y', hover_name='sent', 
                        color=avg_similarities, color_continuous_scale=px.colors.sequential.Plasma,
                        labels={'color': 'Độ tương đồng trung bình'})
    
    fig.update_traces(hovertemplate='<b>%{hovertext}</b><br>Độ tương đồng: %{marker.color:.2f}')
    fig.update_layout(coloraxis_colorbar=dict(
        title="Độ tương đồng",
        tickvals=[0, 1.25, 2.5, 3.75, 5],
        ticktext=["0 (Rất khác biệt)", "1.25", "2.5 (Trung bình)", "3.75", "5 (Rất tương đồng)"]
    ))
    return fig

# Tiêu đề cho ứng dụng
st.title("Phân Tích Tương Đồng Ngữ Nghĩa Mô Hình Bert")

# Tab điều hướng
tab1, tab2, tab3, tab4 = st.tabs(["Nhập từng câu", "Nhập nhiều câu một lần", "Kiểm tra đạo văn", "Tóm tắt văn bản"])

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
                        # Gọi hàm compute_similarity đã chỉnh sửa
                        similarity = compute_similarity(sentences[i], sentences[j])
                        similarities[i][j] = similarity
                        similarities[j][i] = similarity
                    else:
                        # Cùng một câu, coi như tương đồng tối đa = 5.0
                        similarities[i][j] = 5.0

            st.subheader("Điểm Tương Đồng")
            overall_similarity = np.mean(similarities)
            for i in range(num_sentences):
                for j in range(i + 1, num_sentences):
                    st.write(f"Độ tương đồng giữa câu {i + 1} và câu {j + 1}: {similarities[i][j]:.2f}/5")

# Tab 2: Nhập nhiều câu một lần
with tab2:
    st.subheader("Nhập nhiều câu, mỗi câu trên một dòng")
    text_area_input = st.text_area("Nhập các câu", height=150, key="multiple_sentences")
    
    if st.button("So sánh độ tương đồng", key="button_multiple"):
        sentences = [sentence.strip() for sentence in text_area_input.split('\n') if sentence.strip()]

        if len(sentences) >= 2:
            num_sentences = len(sentences)
            similarities = np.zeros((num_sentences, num_sentences))
            
            for i in range(num_sentences):
                for j in range(i, num_sentences):
                    if i != j:
                        similarity = compute_similarity(sentences[i], sentences[j])
                        similarities[i][j] = similarity
                        similarities[j][i] = similarity
                    else:
                        similarities[i][j] = 5.0
            
            st.subheader("Đồng Hồ Đo Lượng Tương Đồng")
            overall_similarity = np.mean(similarities)
            fig_overall_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=overall_similarity,
                title={'text': "Độ tương đồng trung bình"},
                gauge={'axis': {'range': [0, 5]}}
            ))
            st.plotly_chart(fig_overall_gauge)

            for i in range(num_sentences):
                for j in range(i + 1, num_sentences):
                    st.write(f"Độ tương đồng giữa câu {i + 1} và câu {j + 1}: {similarities[i][j]:.2f}/5")

            embeddings = np.array([get_embedding(sentence) for sentence in sentences])
            num_components = min(3, num_sentences)
            pca = PCA(n_components=num_components)
            embeddings_reduced = pca.fit_transform(embeddings)

            df = pd.DataFrame({
                'sent': sentences,
                'x': embeddings_reduced[:, 0],
                'y': embeddings_reduced[:, 1]
            })

            if num_components == 3:
                df['z'] = embeddings_reduced[:, 2]

            avg_similarities = np.mean(similarities, axis=1)
            st.subheader("Biểu Đồ vector Embedding Các Câu")
            fig = create_scatter_plot(df, num_components, avg_similarities)
            st.plotly_chart(fig)
        else:
            st.warning("Vui lòng nhập ít nhất 2 câu.")

# Tab 3: Kiểm tra đạo văn
with tab3:
    st.subheader("Nhập văn bản để kiểm tra đạo văn")
    text_to_check = st.text_area("Nhập văn bản", height=150, key="text_to_check")
    reference_text = st.text_area("Nhập văn bản tham chiếu để so sánh", height=150, key="reference_text")

    if st.button("Kiểm tra đạo văn", key="button_plagiarism"):
        if text_to_check and reference_text:
            similarity = compute_similarity(text_to_check, reference_text)
            st.write(f"Độ tương đồng giữa văn bản gốc và văn bản tham chiếu: {similarity:.2f}/5")
        else:
            st.warning("Vui lòng nhập cả văn bản và văn bản tham chiếu.")

# Tab 4: Tóm tắt văn bản
with tab4:
    st.subheader("Nhập văn bản để tóm tắt")
    text_to_summarize = st.text_area("Nhập văn bản", height=150, key="text_to_summarize")

    if st.button("Tóm tắt", key="button_summary"):
        summary = extractive_summary(text_to_summarize, max_sentences=3)  # max_sentences=3: Sử dụng 3 câu
        col1, col2 = st.columns(2) #  chia hai 
        with col1:
            st.subheader("Nội dung gốc:")
            st.write(text_to_summarize) 
        with col2:# Hiển thị nội dung gốc
            st.subheader("Tóm tắt kết quả:")
            st.write(summary)  # Hiển thị tóm tắt
