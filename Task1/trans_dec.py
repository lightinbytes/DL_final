"""
trans_dec.py — Cấu hình A2: ResNet50 + LSTM Text Encoder + Transformer Decoder

SỬA LỖI CHÍNH:
  - Bỏ CLIPVisionModel → dùng ImageEncoder (ResNet50) từ encoder.py (ĐÚNG yêu cầu đề)
  - Bỏ text embedding riêng → dùng TextEncoder (LSTM) từ encoder.py
  - Giữ nguyên image + text encoder so với A1, CHỈ thay decoder
  - Memory của TransformerDecoder = ghép img_token + txt_token → (B, 2, 512)
  - forward() signature nhất quán với lstm_dec.py và train.py:
      (images, q_ids, q_mask, decoder_input_ids, decoder_mask) → logits (B, seq, vocab)
"""
import math
import torch
import torch.nn as nn
from encoder import ImageEncoder, TextEncoder


# ════════════════════════════════════════════════════════════════════
# Positional Encoding (chuẩn Vaswani 2017)
# ════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


# ════════════════════════════════════════════════════════════════════
# Transformer Decoder (sinh chuỗi autoregressive)
# ════════════════════════════════════════════════════════════════════

class TransformerAnswerDecoder(nn.Module):
    """
    Transformer Decoder sinh chuỗi từ memory (ảnh + câu hỏi).
    - Embedding + PositionalEncoding cho input tokens
    - TransformerDecoderLayer × num_layers với Cross-Attention vào memory
    - Linear head → vocab_size
    """
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model    = d_model
        self.embedding  = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.pos_enc    = PositionalEncoding(d_model, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out  = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Mặt nạ nhân quả: ngăn nhìn token tương lai."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        return mask.masked_fill(mask.bool(), float('-inf'))

    def forward(self,
                decoder_input_ids: torch.Tensor,   # (B, seq)
                memory: torch.Tensor,              # (B, mem_len, d_model)
                decoder_mask: torch.Tensor = None  # (B, seq) — 1=giữ, 0=pad
                ) -> torch.Tensor:                 # → (B, seq, vocab_size)

        tgt = self.pos_enc(
            self.embedding(decoder_input_ids) * math.sqrt(self.d_model)
        )
        seq_len  = tgt.size(1)
        tgt_mask = self._causal_mask(seq_len, tgt.device)

        tgt_key_padding_mask = None
        if decoder_mask is not None:
            tgt_key_padding_mask = (decoder_mask == 0)   # True = bỏ qua

        out    = self.decoder(tgt, memory,
                              tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask)
        return self.fc_out(out)                          # (B, seq, vocab_size)


# ════════════════════════════════════════════════════════════════════
# VQA_Generative_Model — Cấu hình A2
# ════════════════════════════════════════════════════════════════════

class VQA_Generative_Model(nn.Module):
    """
    CẤU HÌNH A2: ResNet50 + LSTM Text Encoder + Transformer Decoder.

    So sánh công bằng với A1 (lstm_dec.py):
      - Giống A1: ImageEncoder (ResNet50), TextEncoder (LSTM), joint_dim=512
      - Khác A1: Decoder là TransformerDecoder thay vì LSTMDecoder

    Memory cho Transformer = [img_token | txt_token] → (B, 2, 512)
    → Cross-Attention trong decoder sẽ attend vào cả ảnh lẫn câu hỏi.
    """
    def __init__(self, vocab_size: int = 64001, joint_dim: int = 512,
                 nhead: int = 8, num_layers: int = 4,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()

        # ── Encoders (chia sẻ với A1) ──────────────────────────────────────
        self.image_encoder = ImageEncoder(fine_tune_last_block=False)   # (B, 2048)
        self.text_encoder  = TextEncoder(vocab_size=vocab_size,
                                         hidden_dim=joint_dim)           # (B, joint_dim)

        # ── Chiếu encodings về joint_dim (memory tokens) ───────────────────
        self.img_proj = nn.Sequential(
            nn.Linear(2048, joint_dim),
            nn.LayerNorm(joint_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(joint_dim, joint_dim),
            nn.LayerNorm(joint_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Transformer Decoder ────────────────────────────────────────────
        self.decoder = TransformerAnswerDecoder(
            vocab_size=vocab_size, d_model=joint_dim,
            nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
        )

    # ── Helper: xây memory từ ảnh + câu hỏi ──────────────────────────────
    def _build_memory(self, images: torch.Tensor,
                      q_ids: torch.Tensor) -> torch.Tensor:
        """
        Trả về memory (B, 2, joint_dim):
          slot 0 → img token
          slot 1 → txt token
        """
        img_feats = self.image_encoder(images)         # (B, 2048)
        txt_feats = self.text_encoder(q_ids)           # (B, joint_dim)
        img_mem   = self.img_proj(img_feats).unsqueeze(1)  # (B, 1, joint_dim)
        txt_mem   = self.txt_proj(txt_feats).unsqueeze(1)  # (B, 1, joint_dim)
        return torch.cat([img_mem, txt_mem], dim=1)        # (B, 2, joint_dim)

    def forward(self,
                images:             torch.Tensor,   # (B, 3, 224, 224)
                q_ids:              torch.Tensor,   # (B, max_q_len)
                q_mask:             torch.Tensor,   # (B, max_q_len) — không dùng ở A2
                decoder_input_ids:  torch.Tensor,   # (B, seq-1)
                decoder_mask:       torch.Tensor,   # (B, seq-1)
                ) -> torch.Tensor:                  # → (B, seq-1, vocab_size)

        memory = self._build_memory(images, q_ids)              # (B, 2, joint_dim)
        logits = self.decoder(decoder_input_ids, memory,
                              decoder_mask)                      # (B, seq-1, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, images: torch.Tensor, q_ids: torch.Tensor,
                 q_mask: torch.Tensor, tokenizer,
                 max_len: int = 20, device: str = 'cuda') -> list:
        """
        Greedy decode cho inference (dùng trong demo_video.ipynb).
        API đồng nhất với VQA_A1_Model.generate().
        """
        self.eval()
        B   = images.size(0)
        bos = tokenizer.cls_token_id or 0
        eos = tokenizer.sep_token_id or 2

        memory    = self._build_memory(images, q_ids)          # (B, 2, joint_dim)
        dec_input = torch.full((B, 1), bos, dtype=torch.long, device=device)

        for _ in range(max_len):
            logits     = self.decoder(dec_input, memory)        # (B, cur, vocab)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            dec_input  = torch.cat([dec_input, next_token], dim=1)
            if (next_token.squeeze(-1) == eos).all():
                break

        generated = dec_input[:, 1:]   # bỏ BOS
        return tokenizer.batch_decode(generated, skip_special_tokens=True)
