# Vietnamese Cuisine Deep Learning Project

Dự án cuối kỳ môn **Học Sâu (Deep Learning)** tập trung vào xử lý hình ảnh và ngôn ngữ tự nhiên trong lĩnh vực ẩm thực Việt Nam. Dự án bao gồm hai nhiệm vụ chính: Trả lời câu hỏi hình ảnh (VQA) và Gợi ý món ăn dựa trên hình ảnh (CBIR).

---

## 📁 Cấu trúc dự án

```text
DL_final/
├── Task1/                      # Bài 1: Visual Question Answering (VQA)
│   ├── dataset.py              # Xử lý dữ liệu và DataLoader
│   ├── encoder.py              # Image & Text Encoders (ResNet50 + PhoBERT)
│   ├── lstm_dec.py             # Kiến trúc A1: LSTM Decoder
│   ├── trans_dec.py            # Kiến trúc A2: Transformer Decoder
│   ├── train.py                # Script huấn luyện chính cho Task 1
│   ├── metrics.py              # Các hàm đánh giá (Recall@K, Accuracy, v.v.)
│   └── notebooks/              # Các file demo và thực nghiệm
├── Task2/                      # Bài 2: Food Recommendation (CBIR)
│   └── Task2_DeepLearning.ipynb # Toàn bộ pipeline Metric Learning & Evaluation
├── requirements.txt            # Danh sách thư viện cần thiết
└── README.md                   # Hướng dẫn sử dụng
```

---

## 🚀 Task 1: Visual Question Answering (VQA)

Hệ thống cho phép người dùng đặt câu hỏi về nội dung của một bức ảnh món ăn và nhận về câu trả lời bằng tiếng Việt.

### Đặc điểm kỹ thuật:
- **Vision Backbone**: ResNet-50 (Pre-trained trên ImageNet).
- **Text Backbone**: PhoBERT (vinai/phobert-base) để embedding câu hỏi tiếng Việt.
- **Decoder**:
  - **A1**: LSTM Decoder với cơ chế Attention.
  - **A2**: Transformer Decoder cho khả năng sinh chuỗi mạnh mẽ hơn.
- **Optimizer**: AdamW với Cosine Annealing Learning Rate.

### Cách chạy:
```bash
python Task1/train.py --model a2 --data_dir data/splits --img_root data/images --batch_size 16 --epochs 15
```

---

## 🍕 Task 2: Food Discovery & Recommendation (CBIR)

Hệ thống gợi ý các món ăn tương đồng dựa trên vector đặc trưng (Embeddings) được huấn luyện bằng kỹ thuật Metric Learning.

### Đặc điểm kỹ thuật:
- **Phương pháp**: Triplet Margin Loss với Hard-Negative Mining.
- **Mô hình so sánh**:
  - **CNN Baseline**: ResNet-50 kết hợp Projection Head.
  - **Transformer Upgrade**: Vision Transformer (ViT-B/16).
- **Tính năng**:
  - **Feature Bank**: Trích xuất và lưu trữ vector đặc trưng của toàn bộ tập dữ liệu.
  - **Visualization**: Trực quan hóa không gian vector bằng **t-SNE** và giải thích mô hình bằng **Attention Maps (ViT)**.
  - **Evaluation**: Đánh giá định lượng qua Recall@K.

---

## 🛠 Cài đặt

Yêu cầu Python 3.8+ và GPU (khuyến khích).

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

Dự án sử dụng bộ dữ liệu **Vietnamese Cuisine v1.0** bao gồm hàng ngàn hình ảnh món ăn Việt Nam đi kèm với các cặp câu hỏi-trả lời mô tả chi tiết.

*Lưu ý: Thư mục dữ liệu `Vietnamese_Cuisine_v1.0` cần được đặt ở thư mục gốc hoặc cấu hình đường dẫn trong script.*