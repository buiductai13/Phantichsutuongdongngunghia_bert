# Quy trình hoạt động của dự án Phân tích Tương đồng Ngữ nghĩa

Tài liệu này mô tả chi tiết các luồng hoạt động chính trong dự án, từ giai đoạn huấn luyện mô hình cho đến khi người dùng cuối sử dụng ứng dụng.

---

## 1. Sơ đồ hoạt động tổng quan

Sơ đồ dưới đây minh họa luồng hoạt động tổng thể của dự án, phân chia rõ ràng giữa hai giai đoạn chính: huấn luyện mô hình (offline) và sử dụng ứng dụng (online).

# Quy trình hoạt động của dự án Phân tích Tương đồng Ngữ nghĩa

Tài liệu này mô tả chi tiết các luồng hoạt động chính trong dự án, từ giai đoạn huấn luyện mô hình cho đến khi người dùng cuối sử dụng ứng dụng.

---

## 1. Sơ đồ hoạt động tổng quan

Sơ đồ dưới đây minh họa luồng hoạt động tổng thể của dự án, phân chia rõ ràng giữa hai giai đoạn chính: huấn luyện mô hình (offline) và sử dụng ứng dụng (online).

```mermaid
graph TD;
    subgraph "Giai đoạn Huấn luyện (Offline)"
        A["Dữ liệu gốc<br/>(Vietnamese STS Benchmark)"] --> B{"Tiền xử lý & Tokenize"};
        C["Mô hình PhoBERT gốc<br/>(vinai/phobert-base)"] --> D{"Fine-tuning"};
        B --> D;
        D --"Đánh giá bằng RMSE"--> E["Lưu mô hình tốt nhất<br/>(modelphobert_similarity_model_2)"];
    end

    subgraph "Giai đoạn Sử dụng (Online - Streamlit App)"
        F["Người dùng nhập văn bản"] --> G{"Ứng dụng Streamlit"};
        E --"1. Tải mô hình"--> G;
        G --"2. Xử lý & Dự đoán"--> H["Tính toán điểm tương đồng<br/>và Tóm tắt văn bản"];
        H --"3. Trả kết quả"--> G;
        G --"4. Hiển thị kết quả"--> I["Giao diện người dùng"];
    end

    style F fill:#cde4ff,stroke:#333,stroke-width:2px
    style I fill:#cde4ff,stroke:#333,stroke-width:2px
```

---

## 2. Chi tiết quá trình Huấn luyện Mô hình (Fine-tuning)

Đây là quy trình chi tiết bên trong giai đoạn huấn luyện. Mục tiêu là dạy cho mô hình `PhoBERT` cách đưa ra một điểm số chính xác để đánh giá sự tương đồng giữa hai câu.

```mermaid
graph TD;
    subgraph "Dữ liệu & Chuẩn bị"
        A["Dữ liệu STS Tiếng Việt<br/>(cặp câu + điểm score)"] --> B["Chia tập Train (80%) và<br/>Tập Evaluation (20%)"];
        B --> C{"Tokenize các cặp câu"};
    end

    subgraph "Quá trình Huấn luyện (Training Loop)"
        D["Tải mô hình PhoBERT-base<br/>(dạng AutoModelForSequenceClassification)"] --> E{"Bắt đầu vòng lặp Huấn luyện (3 epochs)"};
        C --> E;
        E --> F["Huấn luyện trên từng batch<br/>(tính loss MSE)"];
        F --> G{"Đánh giá trên tập Evaluation<br/>(tính metric RMSE)"};
        G -- "RMSE tốt hơn?" --> H["Lưu lại (checkpoint) mô hình<br/>là phiên bản tốt nhất"];
        G -- "Không" --> E;
        H --> E;
    end

    subgraph "Kết quả"
        E -- "Hoàn thành 3 epochs" --> I["Lưu lại mô hình tốt nhất<br/>(modelphobert_similarity_model_2)"];
    end

    style I fill:#d4edda,stroke:#155724
```

**Diễn giải các bước:**
1.  **Chuẩn bị dữ liệu**: Dữ liệu được chia thành hai phần, một để huấn luyện (`train`) và một để đánh giá (`evaluation`) hiệu năng của mô hình sau mỗi chu kỳ.
2.  **Tokenize**: Các câu văn được chuyển đổi thành định dạng số mà mô hình có thể hiểu được.
3.  **Vòng lặp huấn luyện**:
    *   Mô hình học từ dữ liệu trong tập `train`.
    *   Sau mỗi `epoch` (một lượt học qua toàn bộ dữ liệu), mô hình được đánh giá trên tập `evaluation` bằng chỉ số **RMSE (Root Mean Square Error)**.
    *   Nếu mô hình ở `epoch` hiện tại có kết quả tốt hơn (RMSE thấp hơn) so với trước đó, phiên bản này sẽ được lưu lại.
4.  **Kết quả**: Sau khi hoàn tất, mô hình tốt nhất đã được lưu sẽ được sử dụng cho ứng dụng.

---

## 3. Nguyên lý hoạt động của Cosine Similarity

`Cosine Similarity` là một trong những phương pháp nền tảng để đo lường sự tương đồng về ngữ nghĩa. Mặc dù ứng dụng chính (`vn_app.py`) sử dụng mô hình hồi quy đã được huấn luyện để đưa ra điểm số trực tiếp, phiên bản `semantic_similarity_analysis.py` và nguyên lý cốt lõi của mô hình đều dựa trên khái niệm này: **các câu có ngữ nghĩa giống nhau sẽ có các vector biểu diễn nằm gần nhau trong không gian vector.**

Sơ đồ dưới đây giải thích cách hoạt động của phương pháp này.

```mermaid
graph TD;
    subgraph "Đầu vào"
        A["Câu A<br/>'Con mèo đang ngồi trên tấm thảm'"]
        B["Câu B<br/>'Một con mèo ngồi trên thảm'"]
    end

    subgraph "Bước 1: Vector hóa (Embedding)"
        C["Mô hình PhoBERT"]
        A --> C
        B --> C
        C --> D["Vector Embedding A<br/>[0.9, 0.2, ..., 0.7]"]
        C --> E["Vector Embedding B<br/>[0.8, 0.3, ..., 0.6]"]
    end

    subgraph "Bước 2: Tính toán Độ tương đồng Cosine"
        F["Không gian Vector đa chiều"]
        D -- "Vector A" --> F
        E -- "Vector B" --> F
        F -- "Đo góc Theta (θ) giữa hai vector" --> G["Công thức:<br/>sim = cos(θ) = (A · B) / (||A|| ||B||)"]
    end
    
    subgraph "Bước 3: Diễn giải Kết quả"
       G --> H{"Kết quả Cosine Similarity<br/>(Thang điểm -1 đến 1)"};
       H -- "Góc θ nhỏ (gần 0°)<br/>cos(θ) gần 1" --> I["Rất tương đồng"];
       H -- "Góc θ lớn (gần 90°)<br/>cos(θ) gần 0" --> J["Không tương đồng"];
    end

    style C fill:#cde4ff
    style G fill:#fff3cd
```
**Diễn giải các bước:**
1.  **Vector hóa**: Mỗi câu được mô hình ngôn ngữ (ở đây là PhoBERT) chuyển đổi thành một vector số, đại diện cho ý nghĩa của nó.
2.  **Tính toán**: Thay vì đo khoảng cách thông thường, phương pháp này đo **góc (theta - θ)** giữa hai vector.
3.  **Kết quả**:
    *   Nếu góc **θ** nhỏ (hai vector gần như trùng hướng), ý nghĩa của chúng rất giống nhau và điểm tương đồng tiến tới **1**.
    *   Nếu góc **θ** gần 90° (hai vector gần như vuông góc), chúng không có nhiều liên quan về mặt ngữ nghĩa và điểm tương đồng tiến tới **0**.

---

## 2. Chi tiết quá trình Huấn luyện Mô hình (Fine-tuning)

Đây là quy trình chi tiết bên trong giai đoạn huấn luyện. Mục tiêu là dạy cho mô hình `PhoBERT` cách đưa ra một điểm số chính xác để đánh giá sự tương đồng giữa hai câu.

```mermaid
graph TD;
    subgraph "Dữ liệu & Chuẩn bị"
        A["Dữ liệu STS Tiếng Việt<br/>(cặp câu + điểm score)"] --> B["Chia tập Train (80%) và<br/>Tập Evaluation (20%)"];
        B --> C{Tokenize các cặp câu};
    end

    subgraph "Quá trình Huấn luyện (Training Loop)"
        D["Tải mô hình PhoBERT-base<br/>(dạng AutoModelForSequenceClassification)"] --> E{"Bắt đầu vòng lặp Huấn luyện (3 epochs)"};
        C --> E;
        E --> F["Huấn luyện trên từng batch<br/>(tính loss MSE)"];
        F --> G{"Đánh giá trên tập Evaluation<br/>(tính metric RMSE)"};
        G -- "RMSE tốt hơn?" --> H["Lưu lại (checkpoint) mô hình<br/>là phiên bản tốt nhất"];
        G -- "Không" --> E;
        H --> E;
    end

    subgraph "Kết quả"
        E -- "Hoàn thành 3 epochs" --> I["Lưu lại mô hình tốt nhất<br/>(modelphobert_similarity_model_2)"];
    end

    style I fill:#d4edda,stroke:#155724
```

**Diễn giải các bước:**
1.  **Chuẩn bị dữ liệu**: Dữ liệu được chia thành hai phần, một để huấn luyện (`train`) và một để đánh giá (`evaluation`) hiệu năng của mô hình sau mỗi chu kỳ.
2.  **Tokenize**: Các câu văn được chuyển đổi thành định dạng số mà mô hình có thể hiểu được.
3.  **Vòng lặp huấn luyện**:
    *   Mô hình học từ dữ liệu trong tập `train`.
    *   Sau mỗi `epoch` (một lượt học qua toàn bộ dữ liệu), mô hình được đánh giá trên tập `evaluation` bằng chỉ số **RMSE (Root Mean Square Error)**.
    *   Nếu mô hình ở `epoch` hiện tại có kết quả tốt hơn (RMSE thấp hơn) so với trước đó, phiên bản này sẽ được lưu lại.
4.  **Kết quả**: Sau khi hoàn tất, mô hình tốt nhất đã được lưu sẽ được sử dụng cho ứng dụng.

---

## 3. Nguyên lý hoạt động của Cosine Similarity

`Cosine Similarity` là một trong những phương pháp nền tảng để đo lường sự tương đồng về ngữ nghĩa. Mặc dù ứng dụng chính (`vn_app.py`) sử dụng mô hình hồi quy đã được huấn luyện để đưa ra điểm số trực tiếp, phiên bản `semantic_similarity_analysis.py` và nguyên lý cốt lõi của mô hình đều dựa trên khái niệm này: **các câu có ngữ nghĩa giống nhau sẽ có các vector biểu diễn nằm gần nhau trong không gian vector.**

Sơ đồ dưới đây giải thích cách hoạt động của phương pháp này.

```mermaid
graph TD;
    subgraph "Đầu vào"
        A["Câu A<br/>'Con mèo đang ngồi trên tấm thảm'"]
        B["Câu B<br/>'Một con mèo ngồi trên thảm'"]
    end

    subgraph "Bước 1: Vector hóa (Embedding)"
        C["Mô hình PhoBERT"]
        A --> C
        B --> C
        C --> D["Vector Embedding A<br/>[0.9, 0.2, ..., 0.7]"]
        C --> E["Vector Embedding B<br/>[0.8, 0.3, ..., 0.6]"]
    end

    subgraph "Bước 2: Tính toán Độ tương đồng Cosine"
        F["Không gian Vector đa chiều"]
        D -- "Vector A" --> F
        E -- "Vector B" --> F
        F -- "Đo góc Theta (θ) giữa hai vector" --> G["Công thức:<br/>sim = cos(θ) = (A · B) / (||A|| ||B||)"]
    end
    
    subgraph "Bước 3: Diễn giải Kết quả"
       G --> H{"Kết quả Cosine Similarity<br/>(Thang điểm -1 đến 1)"};
       H -- "Góc θ nhỏ (gần 0°)<br/>cos(θ) gần 1" --> I["Rất tương đồng"];
       H -- "Góc θ lớn (gần 90°)<br/>cos(θ) gần 0" --> J["Không tương đồng"];
    end

    style C fill:#cde4ff
    style G fill:#fff3cd
```
**Diễn giải các bước:**
1.  **Vector hóa**: Mỗi câu được mô hình ngôn ngữ (ở đây là PhoBERT) chuyển đổi thành một vector số, đại diện cho ý nghĩa của nó.
2.  **Tính toán**: Thay vì đo khoảng cách thông thường, phương pháp này đo **góc (theta - θ)** giữa hai vector.
3.  **Kết quả**:
    *   Nếu góc **θ** nhỏ (hai vector gần như trùng hướng), ý nghĩa của chúng rất giống nhau và điểm tương đồng tiến tới **1**.
    *   Nếu góc **θ** gần 90° (hai vector gần như vuông góc), chúng không có nhiều liên quan về mặt ngữ nghĩa và điểm tương đồng tiến tới **0**. 
