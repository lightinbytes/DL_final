import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer


class VietnameseVQADataset(Dataset):
    """
    Dataset chuẩn Sinh chuỗi (Autoregressive) cho VQA Paper-Level.
    Sử dụng PhoBERT Tokenizer để số hóa cả Câu hỏi và Câu trả lời.

    Cấu trúc JSON thực tế:
        item['image_path']              → "data/images/am_thuc/vịt_quay/000010.jpg"
        item['qa_pairs'][i]['question'] → câu hỏi
        item['qa_pairs'][i]['answer']   → câu trả lời ngắn (≤ 10 từ)

    Mỗi ảnh có 5 cặp Q&A → dataset sẽ flatten thành N*5 samples.
    """

    def __init__(self, json_path, img_root, max_q_len=30, max_a_len=30, split='train'):
        """
        Args:
            json_path : Đường dẫn tới file train.json / val.json / test.json
            img_root  : Thư mục gốc chứa ảnh, ví dụ: "Vietnamese_Cuise_v1.0/images"
                        Code sẽ ghép: img_root / am_thuc / <món> / <id>.jpg
            max_q_len : Độ dài tối đa câu hỏi (token)
            max_a_len : Độ dài tối đa câu trả lời (token) — dùng answer ngắn ≤ 10 từ
            split     : 'train' | 'val' | 'test' — dùng để chọn transform
        """
        self.img_root  = img_root
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        self.split     = split

        # ── Tokenizer ──────────────────────────────────────────────────────────
        print("Đang nạp PhoBERT Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

        # ── Đọc JSON và flatten Q&A ────────────────────────────────────────────
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Mỗi item có thể có nhiều qa_pairs → flatten thành 1 list phẳng
        # Mỗi sample = (image_path, question, answer)
        self.samples = []
        for item in raw_data:
            image_path = item.get('image_path', '')
            qa_pairs   = item.get('qa_pairs', [])

            for qa in qa_pairs:
                question = qa.get('question', '').strip()
                answer   = qa.get('answer', '').strip()   # Dùng answer ngắn (≤10 từ)
                if question and answer:
                    self.samples.append({
                        'image_path': image_path,
                        'question'  : question,
                        'answer'    : answer,
                    })

        print(f"Loaded {len(raw_data)} ảnh → {len(self.samples)} Q&A samples ({split})")

        # ── Image Transform ────────────────────────────────────────────────────
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225]),
            ])
        else:  # val / test — không augment
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def _resolve_image_path(self, image_path: str) -> str:
        image_path = image_path.replace('\\', '/')

        marker = 'am_thuc/'
        idx = image_path.find(marker)
        if idx != -1:
            relative = image_path[idx:]
            parts = relative.split('/')
            if len(parts) >= 2:
                parts[1] = parts[1].replace(' ', '_')  # "giò thu" → "giò_thu"
            relative = '/'.join(parts)
            return os.path.join(self.img_root, relative)

        return os.path.join(self.img_root, os.path.basename(image_path))

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # ── 1. Xử lý Ảnh ──────────────────────────────────────────────────────
        img_path = self._resolve_image_path(sample['image_path'])
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception:
            # Fallback tensor đen nếu không mở được ảnh
            image = torch.zeros((3, 224, 224))

        # ── 2. Xử lý Câu hỏi ──────────────────────────────────────────────────
        encoded_q = self.tokenizer(
            sample['question'],
            max_length=self.max_q_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        q_ids  = encoded_q['input_ids'].squeeze(0)       # (max_q_len,)
        q_mask = encoded_q['attention_mask'].squeeze(0)  # (max_q_len,)

        # ── 3. Xử lý Câu trả lời ──────────────────────────────────────────────
        encoded_a = self.tokenizer(
            sample['answer'],
            max_length=self.max_a_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        a_ids  = encoded_a['input_ids'].squeeze(0)       # (max_a_len,)
        a_mask = encoded_a['attention_mask'].squeeze(0)  # (max_a_len,)

        return image, q_ids, q_mask, a_ids, a_mask


def get_dataloader(json_path, img_root, batch_size=16, shuffle=True, split='train'):
    """
    Tạo DataLoader cho một split.

    Args:
        json_path  : Đường dẫn file JSON (train/val/test)
        img_root   : Thư mục gốc chứa ảnh (Vietnamese_Cuise_v1.0/images)
        batch_size : Kích thước batch
        shuffle    : True cho train, False cho val/test
        split      : 'train' | 'val' | 'test'
    """
    dataset = VietnameseVQADataset(
        json_path=json_path,
        img_root=img_root,
        split=split,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
    )
