"""
api.py — FastAPI backend cho VQA Demo
"""

import io
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ════════════════════════════════════════════════════════
#   model_a1, model_a2, model_b1, model_b2
#   predict_a1(image, question) -> str
#   predict_a2(image, question) -> str
#   predict_paligemma(model, image, question) -> str
# ════════════════════════════════════════════════════════

print("[api.py] Khởi động FastAPI server...")
print(f"[api.py] model_a1 : {'✅' if 'model_a1' in dir() else '❌ THIẾU'}")
print(f"[api.py] model_a2 : {'✅' if 'model_a2' in dir() else '❌ THIẾU'}")
print(f"[api.py] model_b1 : {'✅' if 'model_b1' in dir() else '❌ THIẾU'}")
print(f"[api.py] model_b2 : {'✅' if 'model_b2' in dir() and model_b2 is not None else '⚠ Chưa load'}")

# ════════════════════════════════════════════════════════
# FASTAPI APP
# ════════════════════════════════════════════════════════
app = FastAPI(title="Vietnamese Food VQA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "VQA API đang chạy!"}


@app.get("/health")
def health():
    """Kiểm tra trạng thái 4 model."""
    return {
        "a1": "model_a1" in dir() and model_a1 is not None,
        "a2": "model_a2" in dir() and model_a2 is not None,
        "b1": "model_b1" in dir() and model_b1 is not None,
        "b2": "model_b2" in dir() and model_b2 is not None,
    }


@app.post("/predict")
async def predict(
    image   : UploadFile = File(...),
    question: str        = Form(...),
):
    """
    Nhận ảnh + câu hỏi → trả về câu trả lời từ 4 model.
    Dùng trực tiếp predict_a1, predict_a2, predict_paligemma
    đã định nghĩa sẵn trong demo_video.ipynb.
    """
    # ── Đọc ảnh ──────────────────────────────────────────
    try:
        img_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Không đọc được ảnh: {str(e)}"}
        )

    if not question.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Câu hỏi không được để trống"}
        )

    # ── Chạy 4 model ─────────────────────────────────────
    # Tái sử dụng hàm predict_* từ notebook, không viết lại
    answers = {}

    try:
        answers["a1"] = predict_a1(pil_image, question)
    except Exception as e:
        answers["a1"] = f"Lỗi A1: {str(e)}"

    try:
        answers["a2"] = predict_a2(pil_image, question)
    except Exception as e:
        answers["a2"] = f"Lỗi A2: {str(e)}"

    try:
        answers["b1"] = predict_paligemma(model_b1, pil_image, question)
    except Exception as e:
        answers["b1"] = f"Lỗi B1: {str(e)}"

    try:
        if "model_b2" in dir() and model_b2 is not None:
            answers["b2"] = predict_paligemma(model_b2, pil_image, question)
        else:
            answers["b2"] = "⚠ Chưa load checkpoint B2"
    except Exception as e:
        answers["b2"] = f"Lỗi B2: {str(e)}"

    return JSONResponse(content={
        "question": question,
        "answers" : answers,
    })
