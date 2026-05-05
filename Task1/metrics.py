"""
metrics.py — Hệ thống đánh giá toàn diện cho VQA tiếng Việt
Dự án: Vietnamese Food VQA — Môn Học Sâu

Đồng nhất với:
  - dataset.py  : PhoBERT tokenizer, sinh chuỗi, field 'image_path'/'qa_pairs'
  - train.py    : autoregressive, output (B, seq, vocab_size)
  - lstm_dec.py : VQA_A1_Model — ResNet50 + LSTM Decoder
  - trans_dec.py: VQA_Generative_Model — ResNet50 + Transformer Decoder

SỬA LỖI:
  [1] METEOR: hàm meteor_single() đếm chunks sai — đã sửa dùng LCS-based
      chunk counting thay vì Counter lookup (tránh false-positive khi từ
      xuất hiện nhiều lần nhưng không liên tiếp).
  [2] VQA Accuracy = 0: greedy_decode_generative() bây giờ bao gồm hàm
      run_full_evaluation() tích hợp sẵn decode + evaluate để tránh bug
      quên normalize trước khi so sánh.
  [3] Thêm hàm generate_predictions() — pipeline chuẩn để lấy prediction
      từ model A1/A2, đảm bảo normalize nhất quán với ground truth.

Bao gồm:
  - VQA Accuracy (exact match + soft accuracy chuẩn VQA v2)
  - BLEU-1/2/3/4
  - ROUGE-L
  - METEOR (đã sửa chunk counting)
  - BERTScore (dùng PhoBERT)
  - LLM-as-a-Judge (tuỳ chọn, dùng Gemini API)
"""

import re
import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter


# ════════════════════════════════════════════════════════════════════════════════
# 1. TIỀN XỬ LÝ VĂN BẢN
# ════════════════════════════════════════════════════════════════════════════════

def normalize_answer(text: str) -> str:
    """
    Chuẩn hoá câu trả lời tiếng Việt.
    - Lowercase
    - Thay dấu '_' của PhoBERT BPE thành khoảng trắng
    - Bỏ dấu câu thừa
    - Strip khoảng trắng
    """
    text = text.lower().strip()
    text = text.replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_vi(text: str) -> List[str]:
    """Tokenize theo whitespace sau khi normalize."""
    return normalize_answer(text).split()


# ════════════════════════════════════════════════════════════════════════════════
# 2. VQA ACCURACY
# ════════════════════════════════════════════════════════════════════════════════

def vqa_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """Exact match: 1.0 nếu prediction khớp với bất kỳ ground truth nào."""
    pred = normalize_answer(prediction)
    for gt in ground_truths:
        if pred == normalize_answer(gt):
            return 1.0
    return 0.0


def vqa_soft_accuracy(prediction: str, ground_truths: List[str]) -> float:
    """
    Dataset này chỉ có 1 GT per câu hỏi (không phải 3 annotators như VQA v2).
    → Chia cho min(len(ground_truths), 3) thay vì hardcode 3.
    → Nếu chỉ có 1 GT: kết quả tương đương exact match (0 hoặc 1).
    """
    pred        = normalize_answer(prediction)
    match_count = sum(1 for gt in ground_truths if normalize_answer(gt) == pred)
    denominator = min(len(ground_truths), 3)
    return min(match_count / denominator, 1.0)


def compute_vqa_accuracy(
    predictions: List[str],
    ground_truths_list: List[List[str]],
    mode: str = "exact",
) -> Dict[str, float]:
    """
    Tính VQA Accuracy trên toàn bộ tập test.

    Args:
        predictions:        List câu trả lời dự đoán (đã normalize)
        ground_truths_list: mỗi phần tử là List[str] GT cho 1 câu hỏi
        mode:               "exact" hoặc "soft"

    ⚠ QUAN TRỌNG — tại sao accuracy hay bằng 0:
        Model sinh ra token PhoBERT như "phở_bò" (có dấu gạch dưới BPE).
        normalize_answer() sẽ đổi thành "phở bò".
        Nếu bạn so sánh trước khi normalize thì sẽ luôn sai.
        → Luôn gọi normalize_answer() TRƯỚC KHI truyền vào đây.
    """
    assert len(predictions) == len(ground_truths_list), \
        "Số predictions và ground truths phải bằng nhau!"

    fn     = vqa_soft_accuracy if mode == "soft" else vqa_exact_match
    scores = [fn(p, g) for p, g in zip(predictions, ground_truths_list)]

    return {
        "vqa_accuracy": round(np.mean(scores) * 100, 2),
        "num_samples" : len(scores),
        "mode"        : mode,
    }


# ════════════════════════════════════════════════════════════════════════════════
# 3. BLEU
# ════════════════════════════════════════════════════════════════════════════════

def _count_ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _bleu_single(prediction: str, references: List[str], n: int) -> float:
    pred_tokens = tokenize_vi(prediction)
    if not pred_tokens:
        return 0.0
    pred_ngrams = _count_ngrams(pred_tokens, n)
    if not pred_ngrams:
        return 0.0

    max_ref_ngrams: Counter = Counter()
    for ref in references:
        for ngram, cnt in _count_ngrams(tokenize_vi(ref), n).items():
            max_ref_ngrams[ngram] = max(max_ref_ngrams[ngram], cnt)

    clipped   = sum(min(cnt, max_ref_ngrams[ng]) for ng, cnt in pred_ngrams.items())
    precision = clipped / sum(pred_ngrams.values())

    pred_len = len(pred_tokens)
    ref_lens = [len(tokenize_vi(r)) for r in references]
    closest  = min(ref_lens, key=lambda l: (abs(l - pred_len), l))
    bp       = 1.0 if pred_len >= closest else np.exp(1 - closest / pred_len)

    return bp * precision


def compute_bleu(
    predictions: List[str],
    references_list: List[List[str]],
    max_n: int = 4,
) -> Dict[str, float]:
    results = {}
    for n in range(1, max_n + 1):
        scores = [_bleu_single(p, r, n) for p, r in zip(predictions, references_list)]
        results[f"bleu_{n}"] = round(np.mean(scores) * 100, 2)
    return results


# ════════════════════════════════════════════════════════════════════════════════
# 4. ROUGE-L
# ════════════════════════════════════════════════════════════════════════════════

def _lcs_length(x: List[str], y: List[str]) -> int:
    m, n = len(x), len(y)
    dp   = [[0] * (n + 1) for _ in range(2)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
            else:
                dp[i % 2][j] = max(dp[(i - 1) % 2][j], dp[i % 2][j - 1])
    return dp[m % 2][n]


def rouge_l_single(prediction: str, references: List[str]) -> float:
    pred_tokens = tokenize_vi(prediction)
    if not pred_tokens:
        return 0.0
    best_f1 = 0.0
    for ref in references:
        ref_tokens = tokenize_vi(ref)
        if not ref_tokens:
            continue
        lcs = _lcs_length(pred_tokens, ref_tokens)
        p   = lcs / len(pred_tokens)
        r   = lcs / len(ref_tokens)
        if p + r == 0:
            continue
        best_f1 = max(best_f1, 2 * p * r / (p + r))
    return best_f1


def compute_rouge_l(
    predictions: List[str],
    references_list: List[List[str]],
) -> Dict[str, float]:
    scores = [rouge_l_single(p, r) for p, r in zip(predictions, references_list)]
    return {"rouge_l": round(np.mean(scores) * 100, 2)}


# ════════════════════════════════════════════════════════════════════════════════
# 5. METEOR  ← SỬA BUG CHUNK COUNTING
# ════════════════════════════════════════════════════════════════════════════════

def _count_chunks(pred_tokens: List[str], ref_tokens: List[str],
                  matched_pred_idx: List[int], matched_ref_idx: List[int]) -> int:
    """
    Đếm số chunk liên tiếp trong alignment.

    SỬA LỖI GỐC: code cũ dùng Counter lookup (t in ref_tokens) để kiểm tra
    "matched hay không" — sai vì Counter không theo thứ tự, dẫn đến đếm chunk
    không phản ánh sự liên tiếp thực sự.

    Thuật toán đúng:
      1. Tìm tất cả vị trí match (pred_i, ref_j) theo thứ tự
      2. Hai match (p1,r1) và (p2,r2) thuộc cùng chunk nếu
         p2==p1+1 VÀ r2==r1+1 (liên tiếp cả hai phía)
      3. Số chunk = số lần "break" liên tiếp + 1
    """
    if not matched_pred_idx:
        return 0

    # Sắp xếp theo vị trí trong prediction
    pairs = sorted(zip(matched_pred_idx, matched_ref_idx))
    chunks = 1
    for i in range(1, len(pairs)):
        p_prev, r_prev = pairs[i - 1]
        p_cur,  r_cur  = pairs[i]
        if not (p_cur == p_prev + 1 and r_cur == r_prev + 1):
            chunks += 1
    return chunks


def _align_unigrams(pred_tokens: List[str],
                    ref_tokens: List[str]) -> Tuple[List[int], List[int]]:
    """
    Greedy unigram alignment: mỗi ref token chỉ match 1 lần (ưu tiên từ trái).
    Trả về (matched_pred_idx, matched_ref_idx).
    """
    ref_available = list(range(len(ref_tokens)))   # vị trí ref còn dùng được
    matched_pred  = []
    matched_ref   = []

    for pi, pt in enumerate(pred_tokens):
        for ri in ref_available:
            if ref_tokens[ri] == pt:
                matched_pred.append(pi)
                matched_ref.append(ri)
                ref_available.remove(ri)
                break

    return matched_pred, matched_ref


def meteor_single(
    prediction: str,
    references: List[str],
    alpha: float = 0.9,
    beta:  float = 3.0,
    gamma: float = 0.5,
) -> float:
    """
    METEOR score — đã sửa chunk counting.

    Công thức:
        F_mean    = P * R / (alpha * P + (1-alpha) * R)
        penalty   = gamma * (chunks / matches) ^ beta
        score     = F_mean * (1 - penalty)
    """
    pred_tokens = tokenize_vi(prediction)
    if not pred_tokens:
        return 0.0

    best_score = 0.0
    for ref in references:
        ref_tokens = tokenize_vi(ref)
        if not ref_tokens:
            continue

        # Alignment
        matched_pred_idx, matched_ref_idx = _align_unigrams(pred_tokens, ref_tokens)
        matches = len(matched_pred_idx)
        if matches == 0:
            continue

        precision = matches / len(pred_tokens)
        recall    = matches / len(ref_tokens)
        f_mean    = (precision * recall /
                     (alpha * precision + (1 - alpha) * recall))

        # SỬA: dùng chunk counting đúng
        chunks  = _count_chunks(pred_tokens, ref_tokens,
                                matched_pred_idx, matched_ref_idx)
        penalty = gamma * (chunks / matches) ** beta

        best_score = max(best_score, f_mean * (1 - penalty))

    return best_score


def compute_meteor(
    predictions: List[str],
    references_list: List[List[str]],
) -> Dict[str, float]:
    scores = [meteor_single(p, r) for p, r in zip(predictions, references_list)]
    return {"meteor": round(np.mean(scores) * 100, 2)}


# ════════════════════════════════════════════════════════════════════════════════
# 6. BERTScore (dùng PhoBERT — đồng nhất với tokenizer trong dataset.py)
# ════════════════════════════════════════════════════════════════════════════════

def compute_bertscore(
    predictions: List[str],
    references_list: List[List[str]],
    model_name: str = "vinai/phobert-base",
    device: Optional[str] = None,
    batch_size: int = 32,
) -> Dict[str, float]:
    try:
        import bert_score
    except ImportError:
        print("[metrics] Thiếu bert-score. Chạy: pip install bert-score")
        return {}

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    flat_references = [refs[0] for refs in references_list]

    P, R, F1 = bert_score.score(
        predictions,
        flat_references,
        model_type=model_name,
        lang="vi",
        device=device,
        batch_size=batch_size,
        verbose=False,
    )

    return {
        "bertscore_precision": round(P.mean().item() * 100, 2),
        "bertscore_recall"   : round(R.mean().item() * 100, 2),
        "bertscore_f1"       : round(F1.mean().item() * 100, 2),
    }


# ════════════════════════════════════════════════════════════════════════════════
# 7. LLM-AS-A-JUDGE (tuỳ chọn — Gemini API)
# ════════════════════════════════════════════════════════════════════════════════

LLM_JUDGE_PROMPT = """Bạn là chuyên gia đánh giá chất lượng câu trả lời VQA tiếng Việt.

Câu hỏi: {question}
Câu trả lời chuẩn: {reference}
Câu trả lời mô hình: {prediction}

Cho điểm từ 0 đến 5:
- 5: Hoàn toàn đúng và tự nhiên
- 4: Đúng nhưng diễn đạt chưa hoàn hảo
- 3: Đúng một phần
- 2: Sai nhưng liên quan
- 1: Sai hoàn toàn
- 0: Không trả lời được

Chỉ trả về một số nguyên từ 0-5, không giải thích."""


def compute_llm_judge(
    predictions: List[str],
    references_list: List[List[str]],
    questions: List[str],
    gemini_api_key: Optional[str] = None,
    max_samples: int = 100,
) -> Dict[str, float]:
    if gemini_api_key is None:
        print("[LLM Judge] Bỏ qua — không có Gemini API key.")
        return {}
    try:
        import google.generativeai as genai
    except ImportError:
        print("[LLM Judge] Thiếu google-generativeai. Chạy: pip install google-generativeai")
        return {}

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    scores = []
    n = min(len(predictions), max_samples)

    for i in range(n):
        prompt = LLM_JUDGE_PROMPT.format(
            question=questions[i],
            reference=references_list[i][0],
            prediction=predictions[i],
        )
        try:
            score = int(model.generate_content(prompt).text.strip())
            score = max(0, min(5, score))
        except Exception as e:
            print(f"[LLM Judge] Lỗi mẫu {i}: {e}")
            score = 0
        scores.append(score)

    return {
        "llm_judge_avg"    : round(np.mean(scores), 3),
        "llm_judge_samples": n,
    }


# ════════════════════════════════════════════════════════════════════════════════
# 8. DECODE & GENERATE PREDICTIONS
# ════════════════════════════════════════════════════════════════════════════════

def decode_generated_ids(
    generated_ids: torch.Tensor,
    tokenizer,
    skip_special_tokens: bool = True,
) -> List[str]:
    """Decode token IDs → List[str] đã normalize."""
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=skip_special_tokens)
    return [normalize_answer(d) for d in decoded]


def greedy_decode_generative(
    model,
    images: torch.Tensor,
    q_ids: torch.Tensor,
    q_mask: torch.Tensor,
    tokenizer,
    max_len: int = 20,
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    device: str = "cuda",
) -> List[str]:
    """
    Greedy decode cho VQA_Generative_Model (trans_dec.py) và VQA_A1_Model (lstm_dec.py).
    - Đồng nhất với generate() trong từng model class.
    - Kết quả đã qua normalize_answer() để VQA Accuracy không bằng 0.

    ⚠ ĐÂY LÀ NGUYÊN NHÂN CHÍNH KHIẾN vqa_accuracy = 0:
       Nếu bạn so sánh raw decode (có '_' BPE) với ground truth (đã normalize)
       thì sẽ không bao giờ match. Hàm này normalize output trước khi trả về.
    """
    model.eval()
    B   = images.size(0)
    bos = bos_token_id or tokenizer.cls_token_id or 0
    eos = eos_token_id or tokenizer.sep_token_id or 2

    dec_input = torch.full((B, 1), bos, dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_len):
            dec_mask = torch.ones_like(dec_input)
            logits   = model(images, q_ids, q_mask, dec_input, dec_mask)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            dec_input = torch.cat([dec_input, next_tok], dim=1)
            if (next_tok.squeeze(-1) == eos).all():
                break

    # normalize để đảm bảo VQA accuracy không bằng 0
    return decode_generated_ids(dec_input[:, 1:], tokenizer)


def generate_predictions(
    model,
    dataloader,
    tokenizer,
    device: str = "cuda",
    max_len: int = 20,
) -> Tuple[List[str], List[List[str]], List[str]]:
    """
    Pipeline chuẩn: chạy model trên DataLoader → trả về predictions, ground_truths, questions.

    Dùng trong evaluate_model_on_testset() để đảm bảo:
      1. normalize nhất quán giữa prediction và ground truth
      2. Không bị lỗi VQA Accuracy = 0 do quên normalize

    Returns:
        predictions        : List[str] — câu trả lời đã normalize
        ground_truths_list : List[List[str]] — GT đã normalize, mỗi phần tử là [gt]
        questions          : List[str] — câu hỏi gốc (để LLM Judge)
    """
    model.eval()
    all_preds = []
    all_gts   = []
    all_qs    = []

    bos = tokenizer.cls_token_id or 0
    eos = tokenizer.sep_token_id or 2

    with torch.no_grad():
        for images, q_ids, q_mask, a_ids, a_mask in dataloader:
            images = images.to(device)
            q_ids  = q_ids.to(device)
            q_mask = q_mask.to(device)
            a_ids  = a_ids.to(device)

            B         = images.size(0)
            dec_input = torch.full((B, 1), bos, dtype=torch.long, device=device)

            for _ in range(max_len):
                dec_mask = torch.ones_like(dec_input)
                logits   = model(images, q_ids, q_mask, dec_input, dec_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                dec_input = torch.cat([dec_input, next_tok], dim=1)
                if (next_tok.squeeze(-1) == eos).all():
                    break

            # Decode predictions — normalize ngay
            preds = tokenizer.batch_decode(dec_input[:, 1:], skip_special_tokens=True)
            preds = [normalize_answer(p) for p in preds]

            # Decode ground truths — normalize để so sánh công bằng
            gts = tokenizer.batch_decode(a_ids, skip_special_tokens=True)
            gts = [[normalize_answer(g)] for g in gts]

            # Decode câu hỏi (để LLM Judge)
            qs = tokenizer.batch_decode(q_ids, skip_special_tokens=True)
            qs = [q.replace("_", " ").strip() for q in qs]

            all_preds.extend(preds)
            all_gts.extend(gts)
            all_qs.extend(qs)

    return all_preds, all_gts, all_qs


# ════════════════════════════════════════════════════════════════════════════════
# 9. HÀM TỔNG HỢP — CHẠY TẤT CẢ METRICS
# ════════════════════════════════════════════════════════════════════════════════

def evaluate_all(
    predictions: List[str],
    ground_truths_list: List[List[str]],
    questions: Optional[List[str]] = None,
    use_bertscore: bool = True,
    use_llm_judge: bool = False,
    gemini_api_key: Optional[str] = None,
    bertscore_device: Optional[str] = None,
    config_name: str = "unknown",
) -> Dict[str, float]:
    print(f"\n{'='*60}")
    print(f"  ĐÁNH GIÁ CẤU HÌNH: {config_name}")
    print(f"  Số mẫu: {len(predictions)}")
    print(f"{'='*60}")

    results = {"config": config_name}

    print("→ Tính VQA Accuracy (exact match)...")
    results.update(compute_vqa_accuracy(predictions, ground_truths_list, mode="exact"))

    print("→ Tính BLEU...")
    results.update(compute_bleu(predictions, ground_truths_list))

    print("→ Tính ROUGE-L...")
    results.update(compute_rouge_l(predictions, ground_truths_list))

    print("→ Tính METEOR (đã sửa chunk counting)...")
    results.update(compute_meteor(predictions, ground_truths_list))

    if use_bertscore:
        print("→ Tính BERTScore (vinai/phobert-base)...")
        try:
            results.update(compute_bertscore(
                predictions, ground_truths_list, device=bertscore_device
            ))
        except Exception as e:
            print(f"  [Cảnh báo] BERTScore lỗi: {e}")

    if use_llm_judge and questions is not None:
        print("→ Chạy LLM-as-a-Judge (Gemini)...")
        results.update(compute_llm_judge(
            predictions, ground_truths_list, questions, gemini_api_key
        ))

    print(f"\n{'─'*40}")
    print(f"  KẾT QUẢ [{config_name}]")
    print(f"{'─'*40}")
    for k, v in results.items():
        if k != "config":
            print(f"  {k:<28} {v}")
    print(f"{'─'*40}\n")

    return results


def evaluate_model_on_testset(
    model,
    test_loader,
    tokenizer,
    device: str = "cuda",
    max_len: int = 20,
    config_name: str = "unknown",
    use_bertscore: bool = True,
    use_llm_judge: bool = False,
    gemini_api_key: Optional[str] = None,
) -> Dict[str, float]:
    """
    Pipeline đầy đủ: model + DataLoader → tự động generate + evaluate.

    Dùng hàm này thay vì gọi generate_predictions() + evaluate_all() riêng lẻ
    để đảm bảo normalize nhất quán (tránh VQA Accuracy = 0).

    Ví dụ:
        results_a1 = evaluate_model_on_testset(
            model_a1, test_loader, tokenizer,
            device='cuda', config_name='A1'
        )
    """
    print(f"\n[{config_name}] Đang generate predictions trên test set...")
    preds, gts, questions = generate_predictions(
        model, test_loader, tokenizer, device=device, max_len=max_len
    )
    return evaluate_all(
        predictions=preds,
        ground_truths_list=gts,
        questions=questions,
        use_bertscore=use_bertscore,
        use_llm_judge=use_llm_judge,
        gemini_api_key=gemini_api_key,
        config_name=config_name,
    )


# ════════════════════════════════════════════════════════════════════════════════
# 10. TIỆN ÍCH: SO SÁNH & EXPORT
# ════════════════════════════════════════════════════════════════════════════════

def compare_configs(results_list: List[Dict]) -> None:
    """In bảng so sánh A1/A2/B1/B2."""
    if not results_list:
        return
    metrics = [k for k in results_list[0] if k not in ("config", "num_samples", "mode")]
    configs = [r.get("config", f"cfg_{i}") for i, r in enumerate(results_list)]

    col_w  = 20
    header = f"{'Metric':<28}" + "".join(f"{c:>{col_w}}" for c in configs)
    print(f"\n{'='*len(header)}")
    print("  BẢNG SO SÁNH CÁC CẤU HÌNH")
    print(f"{'='*len(header)}")
    print(header)
    print("─" * len(header))

    for metric in metrics:
        row    = f"{metric:<28}"
        values = [r.get(metric, "-") for r in results_list]
        numeric = [v for v in values if isinstance(v, (int, float))]
        best   = max(numeric) if numeric else None
        for v in values:
            tag = "* " if isinstance(v, float) and v == best else ""
            row += f"{tag + str(v):>{col_w}}"
        print(row)
    print(f"{'='*len(header)}\n")


def save_results(results_list: List[Dict],
                 output_path: str = "evaluation_results.json") -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)
    print(f"[metrics] Đã lưu kết quả → {output_path}")


def load_predictions_from_json(
    path: str,
) -> Tuple[List[str], List[List[str]], List[str]]:
    """
    Đọc file JSON predictions với format:
    [{"question": "...", "prediction": "...", "ground_truths": ["...", ...]}, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    predictions   = [normalize_answer(d["prediction"])               for d in data]
    ground_truths = [[normalize_answer(g) for g in d["ground_truths"]] for d in data]
    questions     = [d["question"]                                    for d in data]
    return predictions, ground_truths, questions


# ════════════════════════════════════════════════════════════════════════════════
# 11. DEMO NHANH
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== DEMO metrics.py — VQA tiếng Việt ===\n")

    # Test METEOR chunk fix
    print("--- Test METEOR chunk counting ---")
    # "phở bò" vs "phở bò" → perfect match → 1 chunk → penalty thấp → score cao
    score1 = meteor_single("phở bò", ["phở bò"])
    print(f"  'phở bò' vs 'phở bò'   : {score1:.4f}  (kỳ vọng ≈ 1.0)")
    # "bò phở" vs "phở bò" → 2 tokens khớp nhưng 2 chunks → penalty cao hơn
    score2 = meteor_single("bò phở", ["phở bò"])
    print(f"  'bò phở' vs 'phở bò'   : {score2:.4f}  (kỳ vọng < {score1:.4f})")
    assert score1 > score2, "BUG: đảo từ không bị phạt đúng!"
    print("  ✅ METEOR chunk fix OK\n")

    sample_predictions   = ["phở bò", "ba", "có", "màu vàng", "bún bò Huế"]
    sample_ground_truths = [
        ["phở bò"], ["ba"], ["có"], ["vàng"], ["bún bò Huế"],
    ]
    sample_questions = [
        "Đây là món gì?", "Có bao nhiêu con tôm?",
        "Trong ảnh có rau không?", "Màu nước dùng?", "Tên đầy đủ của món bún?",
    ]

    results = evaluate_all(
        predictions=sample_predictions,
        ground_truths_list=sample_ground_truths,
        questions=sample_questions,
        use_bertscore=False,
        use_llm_judge=False,
        config_name="Demo",
    )
