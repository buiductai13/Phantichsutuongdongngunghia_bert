## ⚙️ Quy trình hoạt động dự án

Sơ đồ dưới đây minh họa luồng hoạt động tổng thể của dự án, từ giai đoạn huấn luyện mô hình đến khi người dùng tương tác với ứng dụng.


**Diễn giải chi tiết:**

1.  **Huấn luyện (Offline)**:
    *   **Dữ liệu**: Chúng tôi sử dụng phiên bản tiếng Việt của bộ dữ liệu STS (Semantic Textual Similarity) Benchmark. Mỗi mẫu dữ liệu gồm một cặp câu và một điểm số (từ 0-5) thể hiện mức độ tương đồng.
    *   **Fine-tuning**: Mô hình `vinai/phobert-base` được huấn luyện tiếp (fine-tuned) trên bộ dữ liệu trên. Đây là một tác vụ hồi quy (regression) nhằm mục đích dự đoán điểm tương đồng.
    *   **Đánh giá**: Mô hình được đánh giá sau mỗi epoch bằng chỉ số **RMSE (Root Mean Square Error)**. Mô hình có RMSE thấp nhất trên tập đánh giá sẽ được lưu lại.

2.  **Sử dụng (Online)**:
    *   **Nhập liệu**: Người dùng nhập một hoặc nhiều câu/đoạn văn vào giao diện Streamlit.
    *   **Xử lý**: Ứng dụng nhận dữ liệu, sử dụng tokenizer của PhoBERT để chuẩn bị đầu vào.
    *   **Dự đoán**: Dữ liệu đã xử lý được đưa vào mô hình đã được tinh chỉnh để tính toán điểm tương đồng.
    *   **Hiển thị**: Kết quả (điểm số, văn bản tóm tắt) được hiển thị lại trên giao diện cho người dùng.

---

## 🏋️‍♀️ Dẫn chứng quá trình huấn luyện

Quá trình huấn luyện được thực hiện trong notebook `train/BERT_VN.ipynb`. Dưới đây là các thông tin chính:

- **Mô hình cơ sở**: `vinai/phobert-base`
- **Tác vụ**: Hồi quy (Regression) với `num_labels=1`.
- **Hàm mất mát (Loss function)**: Mặc định là Mean Squared Error (MSE) cho tác vụ hồi quy trong `Trainer`.
- **Chỉ số đánh giá chính**: Root Mean Square Error (RMSE).
- **Các siêu tham số quan trọng**:
    - `learning_rate`: 2e-5
    - `per_device_train_batch_size`: 8
    - `num_train_epochs`: 3
    - `weight_decay`: 0.01

### Kết quả kiểm tra thủ công

Sau khi huấn luyện, mô hình được kiểm tra với các cặp câu mẫu để xác thực:

- **Câu 1**: `đội tuyển việt nam đã giành chiến thắng`
- **Câu 2**: `đội tuyển đem vinh quang về cho việt nam`
- **Kết quả điểm tương đồng**: **~3.68** (Thể hiện sự tương đồng khá cao về mặt ngữ nghĩa)

- **Câu 1**: `đội tuyển việt nam đã giành chiến thắng`
- **Câu 2**: `đội tuyển việt nam đã giành chiến thắng`
- **Kết quả điểm tương đồng**: **~4.90** (Thể hiện sự tương đồng gần như tuyệt đối)

Các kết quả này cho thấy mô hình đã học được cách phân biệt mức độ tương đồng ngữ nghĩa giữa các câu tiếng Việt một cách hiệu quả.

---
