import os
import sys
import argparse
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import AutoTokenizer

# Cố định chuẩn hiển thị UTF-8 trên Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Import chuẩn (Giữ nguyên tên hàm/class để không phá vỡ cấu trúc ngoài)
from src.task1_vqa.dataset import get_dataloader
# Module trans_dec.py sẽ chứa kiến trúc lõi Sinh chuỗi
from src.task1_vqa.models.trans_dec import VQA_Generative_Model 

def train_model():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình VQA - Generative (Sinh chuỗi)")
    parser.add_argument('--data_dir', type=str, required=True, help="Đường dẫn đến thư mục chứa file json")
    parser.add_argument('--img_dir', type=str, required=True, help="Đường dẫn đến thư mục chứa ảnh")
    args = parser.parse_args()

    print("="*60)
    print("KHỞI ĐỘNG HUẤN LUYỆN VQA - MÔ HÌNH SINH CHUỖI (AUTOREGRESSIVE)")
    print("="*60)

    # 1. Đường dẫn động (Đã loại bỏ vocab.json vì dùng PhoBERT Tokenizer)
    TRAIN_PATH = os.path.join(args.data_dir, "train.json")
    VAL_PATH = os.path.join(args.data_dir, "val.json")
    CURRENT_IMG_DIR = args.img_dir 

    # 2. Siêu tham số (Hyperparameters)
    BATCH_SIZE = 16    
    EPOCHS = 15        
    LEARNING_RATE = 2e-5 # LR cho Transformer thường nhỏ hơn LSTM
    
    # 3. Nạp Tokenizer (Thay thế hoàn toàn file vocab tự chế)
    print("Đang nạp PhoBERT Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    PAD_TOKEN_ID = tokenizer.pad_token_id
    VOCAB_SIZE = tokenizer.vocab_size
    print(f"Kích thước từ vựng: {VOCAB_SIZE} | Pad Token ID: {PAD_TOKEN_ID}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị huấn luyện: {device.type.upper()}")

    # 4. Nạp Dữ liệu
    print("Đang nạp dữ liệu...")
    train_loader = get_dataloader(TRAIN_PATH, CURRENT_IMG_DIR, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(VAL_PATH, CURRENT_IMG_DIR, batch_size=BATCH_SIZE, shuffle=False)
    
    # 5. Khởi tạo Mô hình & Hàm tối ưu
    model = VQA_Generative_Model(vocab_size=VOCAB_SIZE).to(device)
    
    # KỸ THUẬT KHOA HỌC 1: Label Smoothing (0.1) & Ignore Padding
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, label_smoothing=0.1)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # KỸ THUẬT KHOA HỌC 2: Cosine Annealing Learning Rate
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    os.makedirs("checkpoints", exist_ok=True)
    
    # Với mô hình sinh chuỗi, ta lưu checkpoint dựa trên Validation Loss thấp nhất, không phải Accuracy
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\n--- EPOCH {epoch+1}/{EPOCHS} ---")
        
        # ==========================================
        # PHASE 1: TRAINING (TEACHER FORCING)
        # ==========================================
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc="Training", leave=False)
        for images, q_ids, q_mask, a_ids, a_mask in loop:
            images = images.to(device)
            q_ids, q_mask = q_ids.to(device), q_mask.to(device)
            a_ids, a_mask = a_ids.to(device), a_mask.to(device)
            
            optimizer.zero_grad()
            
            # KỸ THUẬT KHOA HỌC 3: Cắt lát để ép học (Teacher Forcing)
            decoder_input_ids = a_ids[:, :-1]
            decoder_mask = a_mask[:, :-1]
            labels = a_ids[:, 1:].contiguous() # Dịch phải 1 bước
            
            outputs = model(images, q_ids, q_mask, decoder_input_ids, decoder_mask)
            
            # Tính Loss trên toàn bộ ma trận (Batch * Seq_len, Vocab_size)
            loss = criterion(outputs.view(-1, VOCAB_SIZE), labels.view(-1))
            
            loss.backward()
            
            # Cắt xén Gradient (Gradient Clipping) để chống nổ Gradient trong Transformer
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        
        # Cập nhật Learning Rate sau mỗi Epoch
        scheduler.step()

        # ==========================================
        # PHASE 2: VALIDATION (PERPLEXITY EVALUATION)
        # ==========================================
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, q_ids, q_mask, a_ids, a_mask in val_loader:
                images = images.to(device)
                q_ids, q_mask = q_ids.to(device), q_mask.to(device)
                a_ids, a_mask = a_ids.to(device), a_mask.to(device)
                
                decoder_input_ids = a_ids[:, :-1]
                decoder_mask = a_mask[:, :-1]
                labels = a_ids[:, 1:].contiguous()
                
                outputs = model(images, q_ids, q_mask, decoder_input_ids, decoder_mask)
                loss = criterion(outputs.view(-1, VOCAB_SIZE), labels.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        
        # Tính Perplexity (Độ đo độ mượt mà của ngôn ngữ sinh ra)
        val_perplexity = math.exp(avg_val_loss) if avg_val_loss < 10 else float('inf')

        print(f"Train Loss: {avg_train_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"Val Loss:   {avg_val_loss:.4f} | Val Perplexity: {val_perplexity:.2f}")

        # ==========================================
        # PHASE 3: CHECKPOINTING
        # ==========================================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = f"checkpoints/best_vqa_gen_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, save_path)
            print(f"⭐ Đã lưu Checkpoint (Cải thiện Val Loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    train_model()