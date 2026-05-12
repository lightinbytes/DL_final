import os
import sys
import argparse
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("ignore")
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from src.task1_vqa.dataset import get_dataloader


def train_model():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình VQA - A1 hoặc A2")
    parser.add_argument('--model',      type=str, required=True, choices=['a1', 'a2'],
                        help="Chọn cấu hình: 'a1' (LSTM Decoder) hoặc 'a2' (Transformer Decoder)")
    parser.add_argument('--data_dir',   type=str, required=True,
                        help="Thư mục chứa train.json, val.json")
    parser.add_argument('--img_root',   type=str, required=True,
                        help="Thư mục gốc chứa ảnh")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs',     type=int, default=15)
    parser.add_argument('--lr',         type=float, default=None)
    parser.add_argument('--patience',   type=int, default=3,
                        help="Early stopping: dừng nếu val_loss không giảm sau N epoch")
    args = parser.parse_args()

    # ── Import model ─────────────────────────────────────────────────────────────
    if args.model == 'a1':
        from src.task1_vqa.models.lstm_dec import VQA_A1_Model
    else:
        from src.task1_vqa.models.trans_dec import VQA_Generative_Model

    print("=" * 60)
    if args.model == 'a1':
        print("KHỞI ĐỘNG HUẤN LUYỆN — CẤU HÌNH A1 (LSTM Decoder)")
    else:
        print("KHỞI ĐỘNG HUẤN LUYỆN — CẤU HÌNH A2 (Transformer Decoder)")
    print("=" * 60)

    # ── Đường dẫn ────────────────────────────────────────────────────────────────
    TRAIN_PATH = os.path.join(args.data_dir, "train.json")
    VAL_PATH   = os.path.join(args.data_dir, "val.json")

    # ── Hyperparameters ──────────────────────────────────────────────────────────
    BATCH_SIZE    = args.batch_size
    EPOCHS        = args.epochs
    LEARNING_RATE = args.lr if args.lr else (5e-5 if args.model == 'a1' else 2e-5)
    PATIENCE      = args.patience

    # ── Tokenizer ────────────────────────────────────────────────────────────────
    print("Đang nạp PhoBERT Tokenizer...")
    tokenizer    = AutoTokenizer.from_pretrained("vinai/phobert-base")
    PAD_TOKEN_ID = tokenizer.pad_token_id
    VOCAB_SIZE   = tokenizer.vocab_size
    print(f"Vocab size: {VOCAB_SIZE} | Pad ID: {PAD_TOKEN_ID}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type.upper()}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── DataLoader ───────────────────────────────────────────────────────────────
    print("Đang nạp dữ liệu...")
    train_loader = get_dataloader(TRAIN_PATH, args.img_root,
                                  batch_size=BATCH_SIZE, shuffle=True,  split='train')
    val_loader   = get_dataloader(VAL_PATH,   args.img_root,
                                  batch_size=BATCH_SIZE, shuffle=False, split='val')

    # ── Khởi tạo Model ───────────────────────────────────────────────────────────
    if args.model == 'a1':
        model = VQA_A1_Model(vocab_size=VOCAB_SIZE).to(device)
    else:
        model = VQA_Generative_Model(vocab_size=VOCAB_SIZE).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tham số huấn luyện: {total_params:,}")

    # ── Loss / Optimizer / Scheduler ─────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float('inf')
    no_improve    = 0  # Early stopping counter

    # ── Training Loop ─────────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        print(f"\n{'─'*60}")
        print(f"EPOCH {epoch+1}/{EPOCHS}")
        print(f"{'─'*60}")

        # PHASE 1: TRAINING
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc="Training", leave=False, mininterval=1)
        for images, q_ids, q_mask, a_ids, a_mask in loop:
            images         = images.to(device)
            q_ids, q_mask  = q_ids.to(device),  q_mask.to(device)
            a_ids, a_mask  = a_ids.to(device),  a_mask.to(device)

            optimizer.zero_grad()

            decoder_input_ids = a_ids[:, :-1]
            decoder_mask      = a_mask[:, :-1]
            labels            = a_ids[:, 1:].contiguous()

            outputs = model(images, q_ids, q_mask, decoder_input_ids, decoder_mask)
            loss    = criterion(outputs.reshape(-1, VOCAB_SIZE), labels.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()

        # PHASE 2: VALIDATION
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, q_ids, q_mask, a_ids, a_mask in val_loader:
                images         = images.to(device)
                q_ids, q_mask  = q_ids.to(device), q_mask.to(device)
                a_ids, a_mask  = a_ids.to(device),  a_mask.to(device)

                decoder_input_ids = a_ids[:, :-1]
                decoder_mask      = a_mask[:, :-1]
                labels            = a_ids[:, 1:].contiguous()

                outputs = model(images, q_ids, q_mask, decoder_input_ids, decoder_mask)
                loss    = criterion(outputs.reshape(-1, VOCAB_SIZE), labels.reshape(-1))
                val_loss += loss.item()

        avg_val_loss   = val_loss / len(val_loader)
        val_perplexity = math.exp(min(avg_val_loss, 10))

        print(f"Train Loss : {avg_train_loss:.4f}")
        print(f"Val Loss   : {avg_val_loss:.4f}")
        print(f"Perplexity : {val_perplexity:.2f}")
        print(f"LR         : {scheduler.get_last_lr()[0]:.6f}")

        # PHASE 3: CHECKPOINT + EARLY STOPPING
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve    = 0
            save_path = f"checkpoints/best_vqa_{args.model}.pth"

            # Chi luu phan trainable, bo backbone pretrained (ResNet frozen)
            # Ca A1 lan A2 deu dung ImageEncoder (ResNet50) → key la 'image_encoder.feature_extractor'
            state_dict = {k: v for k, v in model.state_dict().items()
                          if not k.startswith('image_encoder.feature_extractor')}

            torch.save({
                'epoch'               : epoch,
                'model'               : args.model,
                'model_state_dict'    : state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss'       : best_val_loss,
                'vocab_size'          : VOCAB_SIZE,
            }, save_path)
            print(f"Saved checkpoint → {save_path} (Val Loss: {best_val_loss:.4f})")
        else:
            no_improve += 1
            print(f"Không cải thiện ({no_improve}/{PATIENCE})")
            if no_improve >= PATIENCE:
                print(f"Early stopping tại epoch {epoch+1}!")
                break

    print(f"\nHuấn luyện hoàn tất! Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train_model()
