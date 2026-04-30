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
    Sử dụng PhoBERT Tokenizer để số hóa cả Câu hỏi và Câu trả lời dài.
    """
    def __init__(self, json_path, img_dir, max_q_len=30, max_a_len=50):
        self.img_dir = img_dir
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        
        # Load HuggingFace Tokenizer (Tiêu chuẩn khoa học cho Tiếng Việt)
        print("⏳ Đang nạp PhoBERT Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # Chuẩn hóa ảnh cho ViT hoặc ResNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Xử lý Ảnh
        img_name = item['image'] if 'image' in item else item.get('image_path', '')
        # Xử lý đường dẫn động
        img_name = os.path.basename(img_name) 
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            # Fallback tensor đen nếu lỗi ảnh
            image = torch.zeros((3, 224, 224))
            
        # 2. Xử lý Câu hỏi (Source)
        question_text = item['question']
        encoded_q = self.tokenizer(
            question_text,
            max_length=self.max_q_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        q_ids = encoded_q['input_ids'].squeeze(0)
        q_mask = encoded_q['attention_mask'].squeeze(0) # Rất quan trọng cho Transformer
        
        # 3. Xử lý Câu trả lời (Target - Sinh chuỗi)
        # Sử dụng detailed_explanation nếu có để thực hiện Rationales Expansion
        answer_text = item.get('detailed_explanation', item['answer'])
        
        encoded_a = self.tokenizer(
            answer_text,
            max_length=self.max_a_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        a_ids = encoded_a['input_ids'].squeeze(0)
        a_mask = encoded_a['attention_mask'].squeeze(0)
        
        return image, q_ids, q_mask, a_ids, a_mask

def get_dataloader(json_path, img_dir, batch_size=16, shuffle=True):
    dataset = VietnameseVQADataset(json_path, img_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)