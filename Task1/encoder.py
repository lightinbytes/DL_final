import torch
import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):
    """
    Module trích xuất đặc trưng hình ảnh sử dụng ResNet-50.
    Đóng băng các layer đầu để tránh Overfitting và OOM.
    """
    def __init__(self, fine_tune_last_block=False):
        super(ImageEncoder, self).__init__()
        
        # Load pre-trained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Bỏ lớp Linear phân loại cuối cùng
        modules = list(resnet.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)
        
        # Chiến lược đóng băng (Freezing Strategy)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        if fine_tune_last_block:
            for param in self.feature_extractor[-2].parameters():
                param.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        features = torch.flatten(features, 1) # Shape: (batch_size, 2048)
        return features

class TextEncoder(nn.Module):
    """
    Module trích xuất ngữ cảnh câu hỏi tiếng Việt bằng LSTM.
    Phù hợp với Token IDs từ PhoBERT.
    """
    def __init__(self, vocab_size=64001, embed_dim=300, hidden_dim=512, num_layers=1):
        super(TextEncoder, self).__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embed_dim, 
            padding_idx=1 
        )
        
        self.lstm = nn.LSTM(
            input_size=embed_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=False
        )

    def forward(self, question_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(question_ids)
        output, (h_n, c_n) = self.lstm(embedded)
        context_vector = h_n[-1] # Lấy state cuối cùng
        return context_vector