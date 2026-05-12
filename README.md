# Vietnamese Cuisine Deep Learning Project

Dự án cuối kỳ môn **Học Sâu (Deep Learning)** tập trung vào xử lý hình ảnh và ngôn ngữ tự nhiên trong lĩnh vực ẩm thực Việt Nam. Dự án bao gồm hai nhiệm vụ chính: Trả lời câu hỏi hình ảnh (VQA) và Gợi ý món ăn dựa trên hình ảnh (CBIR).

---

## 📁 Cấu trúc dự án

```text
DL_final/
├── Task1/                       # Bài 1: Visual Question Answering (VQA)
│   ├── dataset.py               # Định nghĩa VietnameseVQADataset và DataLoader (Dùng PhoBERT)
│   ├── encoder.py               # Image Encoder (ResNet50) và Text Encoder (LSTM)
│   ├── lstm_dec.py              # Kiến trúc A1: Kết hợp ResNet50 + LSTM Decoder
│   ├── trans_dec.py             # Kiến trúc A2: Kết hợp ResNet50 + Transformer Decoder
│   ├── train.py                 # Script huấn luyện chính (CLI) cho mô hình A1 và A2
│   ├── train_A1_A2.ipynb        # Quy trình train/eval chi tiết cho mô hình A1 & A2
│   ├── train_B1_B2.ipynb        # Thực nghiệm Zero-shot và Fine-tune với PaliGemma
│   ├── metrics.py               # Đánh giá: VQA Accuracy, BLEU, ROUGE, METEOR, BERTScore
│   ├── demo_video.ipynb         # Demo từng kiến trúc (A1, A2, B1, B2) và sử dụng LLM Judge để đánh giá
│   ├── api.py                   # Backend FastAPI phục vụ việc dự đoán (Predict) và Demo
│   └── index.html               # Giao diện Web người dùng (Frontend) để tương tác trực quan
├── Task2/                       # Bài 2: Food Recommendation (CBIR)
│   └── Task2_DeepLearning.ipynb # Toàn bộ pipeline Metric Learning & Evaluation
├── requirements.txt             # Danh sách thư viện cần thiết
└── README.md                    # Hướng dẫn sử dụng
```

---

## Task 1: Visual Question Answering (VQA)

Hệ thống cho phép người dùng đặt câu hỏi về nội dung của một bức ảnh món ăn và nhận về câu trả lời bằng tiếng Việt.

### 1. Đặc điểm kỹ thuật (Technical Stack)

#### Hướng tiếp cận Custom (Mô hình A1 & A2)
* **Vision Backbone**: **ResNet-50** (Pre-trained trên ImageNet) trích xuất đặc trưng hình ảnh dưới dạng vector $2048$ chiều.
* **Text Backbone**: **PhoBERT** (`vinai/phobert-base`) kết hợp **LSTM Encoder** để mã hóa và hiểu ngữ cảnh câu hỏi tiếng Việt.
* **Fusion**: Sử dụng phép nhân **Hadamard Product** để kết hợp đặc trưng ảnh và chữ vào một không gian vector chung ($512$ chiều).
* **Decoder**:
    * **A1 (LSTM Decoder)**: Tích hợp cơ chế **Attention** để tập trung vào các vùng ảnh quan trọng trong quá trình sinh từ.
    * **A2 (Transformer Decoder)**: Tận dụng cơ chế **Self-Attention** và **Cross-Attention** để tối ưu hóa khả năng học mối quan hệ đa phương thức phức tạp.

#### Hướng tiếp cận Foundation Model (Mô hình B1 & B2)
* **Model Base**: **PaliGemma-3B** (`google/paligemma-3b-pt-224`), kiến trúc Vision-Language hiện đại nhất từ Google.
* **Kỹ thuật thực thi**:
    * **B1 (Zero-shot)**: Khai thác khả năng suy luận trực tiếp từ trọng số pre-trained toàn cầu.
    * **B2 (Fine-tuning)**: Áp dụng **LoRA (Low-Rank Adaptation)** để tinh chỉnh mô hình hiệu quả trên tập dữ liệu đặc thù về ẩm thực Việt Nam (phở, bún chả, bánh mì...) với tài nguyên tính toán thấp.

---

### 2. Các kỹ thuật tối ưu hóa sử dụng

* **Teacher Forcing**: Sử dụng câu trả lời chuẩn (Ground Truth) làm đầu vào cho bước thời gian tiếp theo trong quá trình huấn luyện thay vì dùng từ dự đoán của mô hình, giúp tăng tốc độ hội tụ và ổn định lớp Decoder.
* **Sequence Modeling**: Chiến lược **Autoregressive Generation** với các token đặc biệt `<bos>` và `<eos>` để kiểm soát chặt chẽ quá trình bắt đầu và kết thúc chuỗi.
* **Optimizer**: **AdamW** kết hợp **Weight Decay** giúp kiểm soát hiện tượng Overfitting hiệu quả.
* **Scheduler**: **Cosine Annealing Learning Rate** điều chỉnh tốc độ học giảm dần theo hàm Cosine, giúp mô hình đạt điểm hội tụ tối ưu ở giai đoạn cuối.
* **Đánh giá đa chiều**: Hệ thống đánh giá toàn diện gồm: **VQA Accuracy**, **BLEU (1-4)**, **ROUGE-L**, **METEOR** và đặc biệt là **BERTScore** (dựa trên PhoBERT) để đo lường độ chính xác về mặt ngữ nghĩa thực tế.

---.

### Cách chạy:
```bash
python Task1/train.py --model a2 --data_dir data/splits --img_root data/images --batch_size 16 --epochs 15
```

---

## Task 2: Food Discovery & Recommendation (CBIR)

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
