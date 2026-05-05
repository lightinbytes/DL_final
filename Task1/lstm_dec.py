import torch
import torch.nn as nn
import math
from encoder import ImageEncoder, TextEncoder


class PositionalEncoding(nn.Module):
    """Positional Encoding chuẩn (Attention Is All You Need)."""
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class FusionLayer(nn.Module):
    """
    Trộn đặc trưng ảnh và câu hỏi bằng Hadamard Product.
    Chiếu cả hai về cùng joint_dim trước khi nhân element-wise.
    """
    def __init__(self, img_feat_dim=2048, txt_feat_dim=512,
                 joint_dim=512, dropout=0.5):
        super().__init__()
        self.img_proj = nn.Sequential(
            nn.Linear(img_feat_dim, joint_dim),
            nn.LayerNorm(joint_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(txt_feat_dim, joint_dim),
            nn.LayerNorm(joint_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, img_feats, txt_feats):
        """
        Args:
            img_feats : (B, img_feat_dim)
            txt_feats : (B, txt_feat_dim)
        Returns:
            fused     : (B, joint_dim)
        """
        v_i = self.img_proj(img_feats)   # (B, joint_dim)
        v_q = self.txt_proj(txt_feats)   # (B, joint_dim)
        return v_i * v_q                  # Hadamard product


class LSTMDecoder(nn.Module):
    """
    LSTM Decoder sinh chuỗi (Autoregressive).
    Nhận memory vector từ FusionLayer và decoder_input_ids (teacher forcing).
    """
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.pos_enc    = PositionalEncoding(embed_dim, dropout)
        self.lstm       = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out     = nn.Linear(hidden_dim, vocab_size)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, decoder_input_ids, memory, h0=None, c0=None):
        """
        Args:
            decoder_input_ids : (B, seq_len) — token IDs đầu vào (teacher forcing)
            memory            : (B, hidden_dim) — context từ FusionLayer
            h0, c0            : hidden state khởi tạo (None → dùng memory)
        Returns:
            logits : (B, seq_len, vocab_size)
        """
        # Khởi tạo hidden state từ memory nếu chưa có
        if h0 is None:
            # Mở rộng memory thành (num_layers, B, hidden_dim)
            h0 = memory.unsqueeze(0).expand(self.lstm.num_layers, -1, -1).contiguous()
            c0 = torch.zeros_like(h0)

        embeds = self.pos_enc(self.embedding(decoder_input_ids))  # (B, seq, embed_dim)
        out, _ = self.lstm(embeds, (h0, c0))                      # (B, seq, hidden_dim)
        logits  = self.fc_out(self.dropout(out))                  # (B, seq, vocab_size)
        return logits


class VQA_A1_Model(nn.Module):
    """
    CẤU HÌNH A1: ResNet Image Encoder + LSTM Text Encoder + LSTM Decoder.

    forward() nhận đủ 5 tham số để tương thích với train.py:
        images, q_ids, q_mask, decoder_input_ids, decoder_mask
    """
    def __init__(self, vocab_size=64001, joint_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder(fine_tune_last_block=False)   # output: (B, 2048)
        self.text_encoder  = TextEncoder(vocab_size=vocab_size,
                                         hidden_dim=joint_dim)           # output: (B, joint_dim)
        self.fusion        = FusionLayer(img_feat_dim=2048,
                                         txt_feat_dim=joint_dim,
                                         joint_dim=joint_dim)
        self.decoder       = LSTMDecoder(vocab_size=vocab_size,
                                          hidden_dim=joint_dim)

    def forward(self, images, q_ids, q_mask,
                decoder_input_ids, decoder_mask):
        """
        FIX LỖI 3 & 4:
        - Nhận đủ 5 tham số như train.py yêu cầu
        - Output shape: (B, seq_len, vocab_size) → tương thích CrossEntropyLoss sinh chuỗi

        Args:
            images            : (B, 3, 224, 224)
            q_ids             : (B, max_q_len)     — PhoBERT token IDs của câu hỏi
            q_mask            : (B, max_q_len)     — attention mask câu hỏi (không dùng trong A1)
            decoder_input_ids : (B, seq_len-1)     — teacher forcing input (a_ids[:, :-1])
            decoder_mask      : (B, seq_len-1)     — mask câu trả lời (không dùng trong LSTM)
        Returns:
            logits            : (B, seq_len-1, vocab_size)
        """
        img_feats  = self.image_encoder(images)              # (B, 2048)
        txt_feats  = self.text_encoder(q_ids)                # (B, joint_dim)
        memory     = self.fusion(img_feats, txt_feats)       # (B, joint_dim)
        logits     = self.decoder(decoder_input_ids, memory) # (B, seq_len-1, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, images, q_ids, q_mask, tokenizer,
                 max_len=20, device='cuda'):
        """
        Greedy decode để inference (dùng trong demo_video.ipynb).

        Returns:
            List[str] câu trả lời đã decode
        """
        self.eval()
        B   = images.size(0)
        bos = tokenizer.cls_token_id or 0
        eos = tokenizer.sep_token_id or 2

        img_feats = self.image_encoder(images)
        txt_feats = self.text_encoder(q_ids)
        memory    = self.fusion(img_feats, txt_feats)

        # Khởi tạo hidden state
        h0 = memory.unsqueeze(0).expand(
            self.decoder.lstm.num_layers, -1, -1).contiguous()
        c0 = torch.zeros_like(h0)

        dec_input = torch.full((B, 1), bos, dtype=torch.long, device=device)

        for _ in range(max_len):
            logits = self.decoder(dec_input, memory, h0, c0)  # (B, cur, vocab)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            dec_input  = torch.cat([dec_input, next_token], dim=1)
            if (next_token.squeeze(-1) == eos).all():
                break

        generated = dec_input[:, 1:]  # bỏ BOS
        return tokenizer.batch_decode(generated, skip_special_tokens=True)
