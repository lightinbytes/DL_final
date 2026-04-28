import torch
import torch.nn as nn
# Sử dụng Import tuyệt đối chuẩn
from src.task1_vqa.models.encoder import ImageEncoder, TextEncoder

class FusionDecoder(nn.Module):
    """
    Module giải mã sử dụng Joint Embedding Space và Hadamard Product.
    """
    def __init__(self, img_feat_dim=2048, txt_feat_dim=512, joint_dim=1024, num_classes=500, dropout=0.5):
        super(FusionDecoder, self).__init__()
        
        self.img_proj = nn.Sequential(
            nn.Linear(img_feat_dim, joint_dim),
            nn.BatchNorm1d(joint_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.txt_proj = nn.Sequential(
            nn.Linear(txt_feat_dim, joint_dim),
            nn.BatchNorm1d(joint_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(joint_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, img_features: torch.Tensor, txt_features: torch.Tensor) -> torch.Tensor:
        v_i = self.img_proj(img_features)
        v_q = self.txt_proj(txt_features)
        
        # Trộn đặc trưng bằng Tích Hadamard
        h = v_i * v_q 
        
        logits = self.classifier(h)
        return logits

class VQA_A1_Model(nn.Module):
    """
    MÔ HÌNH TOÀN DIỆN CHO CẤU HÌNH A1
    """
    def __init__(self, num_classes: int, vocab_size=64001):
        super(VQA_A1_Model, self).__init__()
        self.image_encoder = ImageEncoder(fine_tune_last_block=False)
        self.text_encoder = TextEncoder(vocab_size=vocab_size)
        self.decoder = FusionDecoder(num_classes=num_classes)

    def forward(self, images: torch.Tensor, question_ids: torch.Tensor) -> torch.Tensor:
        img_feats = self.image_encoder(images)
        txt_feats = self.text_encoder(question_ids)
        logits = self.decoder(img_feats, txt_feats)
        return logits