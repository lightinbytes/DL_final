import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from src.task1_vqa.dataset import get_dataloader
from src.task1_vqa.models.lstm_dec import VQA_A1_Model

def train_model():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình VQA - Cấu hình A1")
    parser.add_argument('--data_dir', type=str, required=True, help="Đường dẫn đến thư mục chứa file json")
    parser.add_argument('--img_dir', type=str, required=True, help="Đường dẫn đến thư mục chứa ảnh")
    args = parser.parse_args()

    print("="*50)
    print("KHỞI ĐỘNG QUÁ TRÌNH HUẤN LUYỆN VQA - CẤU HÌNH A1")
    print("="*50)

    # Khởi tạo đường dẫn động
    TRAIN_PATH = os.path.join(args.data_dir, "train.json")
    VAL_PATH = os.path.join(args.data_dir, "val.json")
    VOCAB_PATH = os.path.join(args.data_dir, "answer_vocab.json")
    CURRENT_IMG_DIR = args.img_dir 

    BATCH_SIZE = 16    
    EPOCHS = 10        
    LEARNING_RATE = 5e-5 
    
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    NUM_CLASSES = len(vocab)
    print(f"Số lượng nhãn (Classes) cần dự đoán: {NUM_CLASSES}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị huấn luyện: {device.type.upper()}")

    print("Đang nạp dữ liệu...")
    train_loader = get_dataloader(TRAIN_PATH, CURRENT_IMG_DIR, VOCAB_PATH, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(VAL_PATH, CURRENT_IMG_DIR, VOCAB_PATH, batch_size=BATCH_SIZE, shuffle=False)
    
    model = VQA_A1_Model(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    os.makedirs("checkpoints", exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\n--- EPOCH {epoch+1}/{EPOCHS} ---")
        
        # --- PHASE 1: TRAINING ---
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        loop = tqdm(train_loader, desc="Training", leave=False)
        for images, questions, labels in loop:
            images, questions, labels = images.to(device), questions.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, questions)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # --- PHASE 2: VALIDATION ---
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        
        with torch.no_grad():
            for images, questions, labels in val_loader:
                images, questions, labels = images.to(device), questions.to(device), labels.to(device)
                
                outputs = model(images, questions)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val

        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # --- PHASE 3: CHECKPOINTING ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"checkpoints/best_vqa_a1_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, save_path)
            print(f"Đã lưu Checkpoint mới tại: {save_path} (Val Acc: {val_acc:.2f}%)")

if __name__ == "__main__":
    train_model()