import math
import torch
import torch.nn as nn
from transformers import CLIPVisionModel

class PositionalEncoding(nn.Module):
    """
    Kỹ thuật Tiêm vị trí (Positional Encoding).
    Transformer không có khái niệm về thứ tự từ, ta phải dùng sóng Sine/Cosine 
    để nhúng vị trí của từ vào vector đặc trưng. (Chuẩn bài báo Attention Is All You Need).
    """
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class VQA_Generative_Model(nn.Module):
    """
    MÔ HÌNH SINH CHUỖI ĐA PHƯƠNG THỨC (AUTOREGRESSIVE MULTIMODAL VQA)
    Sử dụng CLIP (ViT) làm Mắt, và Transformer Decoder làm Não.
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1):
        super(VQA_Generative_Model, self).__init__()
        
        self.d_model = d_model
        
        # ==========================================
        # 1. VISION ENCODER (Hệ thị giác - CLIP)
        # ==========================================
        print("⚙️ Đang khởi tạo Hệ thị giác CLIP (ViT-B/32)...")
        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Đóng băng (Freeze) các lớp dưới của CLIP để tiết kiệm VRAM và giữ nguyên tri thức ảnh
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        # CLIP ViT-B/32 trả về vector kích thước 768. Ta chiếu nó về d_model (512)
        self.vision_proj = nn.Linear(768, d_model)
        self.vision_norm = nn.LayerNorm(d_model)

        # ==========================================
        # 2. TEXT EMBEDDING (Hệ ngôn ngữ)
        # ==========================================
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # ==========================================
        # 3. TRANSFORMER DECODER (Khối Sinh Chuỗi Tự Hồi Quy)
        # ==========================================
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True,
            activation="gelu" # Chuẩn khoa học của Generative AI
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Lớp phân loại để xuất ra xác suất của từng từ trong từ điển
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        """
        Mặt nạ Causal (Causal Mask). 
        Ngăn không cho mô hình "nhìn trộm" các từ trong tương lai khi đang học sinh chuỗi.
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, images, q_ids, q_mask, a_ids, a_mask):
        """
        Luồng đi của dữ liệu (Data Pipeline) trong kiến trúc.
        """
        device = images.device
        batch_size = images.size(0)

        # --- BƯỚC 1: XỬ LÝ ẢNH (Tạo Visual Tokens) ---
        # Đầu ra: (batch, num_patches, 768). Với CLIP patch32, ảnh 224x224 có 50 patches.
        vision_outputs = self.vision_encoder(pixel_values=images)
        img_feats = vision_outputs.last_hidden_state 
        
        # Chiếu về d_model và chuẩn hóa -> (batch, 50, d_model)
        img_embeds = self.vision_norm(self.vision_proj(img_feats))
        
        # --- BƯỚC 2: XỬ LÝ CÂU HỎI (Tạo Question Tokens) ---
        # Nhân căn bậc 2 của d_model để cân bằng variance trước khi cộng Positional (Chuẩn bài báo)
        q_embeds = self.text_embedding(q_ids) * math.sqrt(self.d_model)
        q_embeds = self.pos_encoder(q_embeds)
        
        # --- BƯỚC 3: TẠO BỘ NHỚ ĐA PHƯƠNG THỨC (Multimodal Memory) ---
        # Ghép Visual Tokens và Question Tokens lại thành 1 chuỗi ký ức duy nhất
        # memory shape: (batch, 50 + seq_q, d_model)
        memory = torch.cat([img_embeds, q_embeds], dim=1)
        
        # Tạo mặt nạ cho bộ nhớ: Ảnh không có padding (toàn False), Câu hỏi thì dùng q_mask (đảo ngược True/False)
        # Trong PyTorch: True nghĩa là "Bỏ qua vị trí này" (Padding)
        img_padding_mask = torch.zeros((batch_size, img_embeds.size(1)), dtype=torch.bool, device=device)
        q_padding_mask = (q_mask == 0) 
        memory_key_padding_mask = torch.cat([img_padding_mask, q_padding_mask], dim=1)

        # --- BƯỚC 4: XỬ LÝ CÂU TRẢ LỜI ĐẦU VÀO (Teacher Forcing) ---
        tgt_embeds = self.text_embedding(a_ids) * math.sqrt(self.d_model)
        tgt_embeds = self.pos_encoder(tgt_embeds)
        
        # Mặt nạ nhân quả (Tránh nhìn tương lai) và Mặt nạ padding
        seq_a_len = tgt_embeds.size(1)
        tgt_mask = self.generate_square_subsequent_mask(seq_a_len, device)
        tgt_key_padding_mask = (a_mask == 0)

        # --- BƯỚC 5: GIẢI MÃ (Decoding) ---
        # Bản giao hưởng của Cross-Attention: tgt (Câu trả lời) liên tục truy vấn memory (Ảnh + Câu hỏi)
        out = self.transformer_decoder(
            tgt=tgt_embeds, 
            memory=memory, 
            tgt_mask=tgt_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask, 
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # --- BƯỚC 6: XUẤT XÁC SUẤT ---
        logits = self.fc_out(out) # shape: (batch, seq_a_len, vocab_size)
        
        return logits