import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

class VietnameseVQADataset(Dataset):
    def __init__(self, data_path, img_dir, vocab_path, max_length=20, transform=None):
        """
        Khởi tạo Dataset cho mô hình VQA Hướng A.
        """
        # Đọc dữ liệu và từ điển nhãn
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        self.img_dir = img_dir
        self.max_length = max_length

        # Khởi tạo PhoBERT Tokenizer cho tiếng Việt
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

        # Cấu hình tiền xử lý ảnh (Resize và Normalize theo chuẩn ImageNet)
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # Trải phẳng dữ liệu (1 ảnh : 1 câu hỏi : 1 câu trả lời)
        self.flat_data = []
        for item in self.data:
            # Điều chỉnh đuôi file tùy thuộc vào format gốc (.jpg hoặc .png)
            img_name = f"{item['image_id']}.jpg" 
            img_path = os.path.join(img_dir, item.get('category', 'am_thuc'), img_name)
            
            for qa in item['qa_pairs']:
                self.flat_data.append({
                    "image_path": img_path,
                    "question": qa['question'],
                    "answer": qa['answer']
                })

    def __len__(self):
        return len(self.flat_data)

    def __getitem__(self, idx):
        item = self.flat_data[idx]

        # 1. Trích xuất Tensor Ảnh
        try:
            image = Image.open(item['image_path']).convert('RGB')
            img_tensor = self.transform(image)
        except Exception:
            # Trả về ma trận 0 nếu file ảnh bị lỗi
            img_tensor = torch.zeros((3, 224, 224))

        # 2. Trích xuất Tensor Câu Hỏi (Đã padding)
        encoded_q = self.tokenizer(
            item['question'],
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        question_tensor = encoded_q['input_ids'].squeeze(0)

        # 3. Trích xuất Tensor Nhãn (Chuyển chuỗi Text thành ID nguyên)
        ans_text = item['answer'].strip().lower()
        # Mặc định gán nhãn 0 nếu từ khóa không tồn tại trong từ điển
        label_id = self.vocab.get(ans_text, 0)
        label_tensor = torch.tensor(label_id, dtype=torch.long)

        return img_tensor, question_tensor, label_tensor

# Hàm tiện ích để nạp dữ liệu thẳng vào mô hình
def get_dataloader(data_path, img_dir, vocab_path, batch_size=32, shuffle=True):
    dataset = VietnameseVQADataset(data_path, img_dir, vocab_path)
    # Lưu ý: Trên Windows để num_workers=0 để tránh lỗi đa luồng, dùng trên Colab có thể đặt là 2
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader