# Semantic Similarity Analysis for Vietnamese using BERT
# 🎯 Mục Tiêu Dự Án
Dự án nhằm xây dựng một hệ thống thông minh có khả năng hiểu và đo lường mức độ tương đồng ngữ nghĩa giữa hai câu tiếng Việt – một bài toán cốt lõi trong xử lý ngôn ngữ tự nhiên. Để hiện thực điều đó, nhóm đã lựa chọn và triển khai các công nghệ tiên tiến như sau:

✨ Ứng dụng mô hình PhoBERT-base – mô hình ngôn ngữ tối ưu cho tiếng Việt được phát triển bởi VinAI – để mã hóa câu văn thành các vector ngữ nghĩa trong không gian đa chiều.

🧠 Biến câu thành vector và đo lường ngữ nghĩa bằng Cosine Similarity, từ đó tính được mức độ tương đồng giữa các câu một cách chính xác và hiệu quả.

🛠️ Xây dựng giao diện web trực quan với Streamlit, giúp người dùng có thể dễ dàng nhập hai câu và nhận được kết quả phân tích ngữ nghĩa ngay lập tức, phục vụ cho các ứng dụng như:

+ Phát hiện đạo văn

+ Tìm kiếm ngữ nghĩa

+ Chatbot hiểu ngữ cảnh

+ Tóm tắt văn bản theo ý nghĩa.

 Mục tiêu không chỉ là tạo ra một mô hình chính xác, mà còn là một công cụ trực quan, dễ sử dụng, có khả năng hiểu sâu tiếng Việt, hướng tới các ứng dụng thực tiễn trong giáo dục, truyền thông và công nghệ thông minh.

# 🔮 Hướng Phát Triển Tương Lai

- Mở rộng khả năng tóm tắt văn bản tiếng Việt.

- Tích hợp mô hình vào API Flask hoặc RESTful API.

- Ứng dụng vào các hệ thống giáo dục, công cụ phát hiện đạo văn, và tìm kiếm văn bản thông minh.

# Phân Tích Tương Đồng Ngữ Nghĩa Tiếng Việt sử dụng PhoBERT

Dự án này là một ứng dụng web được xây dựng bằng Streamlit để phân tích và tính toán độ tương đồng về mặt ngữ nghĩa giữa các câu hoặc đoạn văn bản tiếng Việt. Ứng dụng sử dụng mô hình `PhoBERT` đã được tinh chỉnh (fine-tuned) cho tác vụ hồi quy tương đồng câu (Sentence Similarity Regression).

## ✨ Chức năng chính

- **So sánh độ tương đồng câu**: Người dùng có thể nhập hai hoặc nhiều câu để nhận điểm tương đồng trên thang điểm từ 0 đến 5.
- **Phát hiện sao chép/đạo văn**: So sánh hai đoạn văn bản để đánh giá mức độ tương đồng, hữu ích cho việc phát hiện nội dung sao chép.
- **Tóm tắt văn bản**: Cung cấp một bản tóm tắt ngắn gọn cho một đoạn văn bản dài dựa trên thuật toán trích xuất (extractive summarization) bằng cách xác định các câu quan trọng nhất.


## 📂 Cấu trúc thư mục

```
phantichsutuongdongngunghia_bert/
│
├── data/
│   └── (Chứa dữ liệu huấn luyện, ví dụ: processed_data_vn.csv)
│
├── model/
│   ├── modelphobert_similarity_model_2/
│   │   └── (Chứa các file của mô hình đã được tinh chỉnh)
│   └── phobert_similarity_model/
│       └── (Một phiên bản khác của mô hình)
│
├── src/
│   ├── vn_app.py                   # Entry point chính của ứng dụng Streamlit
│   ├── app.py                      # Một phiên bản khác của ứng dụng
│   └── semantic_similarity_analysis.py # Phiên bản ứng dụng với tính năng visualization
│
├── train/
│   └── BERT_VN.ipynb               # Jupyter Notebook cho việc huấn luyện mô hình
│
└── README.md                       # File này
```

## 🚀 Cài đặt và Chạy dự án

### 1. Yêu cầu hệ thống

- Python 3.8+
- Git

### 2. Cài đặt

1.  **Clone repository về máy:**
    ```bash
    git clone <URL_CUA_REPOSITORY>
    cd phantichsutuongdongngunghia_bert
    ```

2.  **Tạo và kích hoạt môi trường ảo (khuyến nghị):**
    ```bash
    python -m venv .venv
    # Trên Windows
    .venv\Scripts\activate
    # Trên macOS/Linux
    source .venv/bin/activate
    ```

3.  **Cài đặt các thư viện cần thiết:**
    Tạo một file `requirements.txt` với nội dung sau:
    ```txt
    streamlit
    torch
    transformers
    scikit-learn
    pandas
    numpy
    plotly
    seaborn
    pyarrow
    ```
    Sau đó chạy lệnh:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Tải mô hình

Đảm bảo rằng thư mục `model/modelphobert_similarity_model_2` chứa đầy đủ các file của mô hình đã được fine-tuned từ notebook huấn luyện. Nếu chưa có, bạn cần chạy notebook `train/BERT_VN.ipynb` để huấn luyện và lưu lại mô hình.

### 4. Chạy ứng dụng

Sau khi cài đặt xong, chạy ứng dụng Streamlit bằng lệnh sau:

```bash
streamlit run src/vn_app.py
```

Ứng dụng sẽ được mở trên trình duyệt của bạn.

## 🏋️‍♀️ Huấn luyện mô hình

Toàn bộ quy trình huấn luyện được mô tả trong file `train/BERT_VN.ipynb`. Về cơ bản, quy trình bao gồm:
1.  **Tải dữ liệu**: Sử dụng bộ dữ liệu Vietnamese STS Benchmark.
2.  **Mô hình cơ sở**: `vinai/phobert-base`.
3.  **Tinh chỉnh**: Tinh chỉnh mô hình cho tác vụ hồi quy (regression) để dự đoán điểm tương đồng giữa hai câu.
4.  **Lưu mô hình**: Mô hình sau khi huấn luyện được lưu lại để ứng dụng có thể sử dụng. 

# 📬 Liên Hệ
📧 Email: buiductaicnnt@gmail.com
📍HCM
📘 GitHub: buiductai13
