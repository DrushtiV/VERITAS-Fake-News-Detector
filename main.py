import os, re, joblib, numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure the preprocessor is importable by joblib during model reload
from app.preprocessor import clean_text  # noqa: F401 — required for joblib deserialization

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "model_pipeline.pkl"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="VERITAS — Fake News Detector API",
    description="TF-IDF + Logistic Regression fake news classifier. Built by Drushti Vagal.",
    version="1.0.0",
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model not found. Run: python model/train.py")
        _pipeline = joblib.load(str(MODEL_PATH))
    return _pipeline


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000,
        example="Scientists at CERN announced new findings about the Higgs boson.")

class PredictResponse(BaseModel):
    label: str
    confidence: float
    real_probability: float
    fake_probability: float
    top_fake_indicators: list[str]
    top_real_indicators: list[str]
    word_count: int
    warning: str | None


def get_top_keywords(pipeline, text, n=5):
    try:
        vec = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]
        mat = vec.transform([text])
        names = np.array(vec.get_feature_names_out())
        vals = mat.toarray()[0]
        coef = clf.coef_[0]
        scores = coef * vals
        fake_idx = np.argsort(scores)[::-1][:n*3]
        real_idx = np.argsort(scores)[:n*3]
        fake_words = [names[i] for i in fake_idx if vals[i] > 0][:n]
        real_words = [names[i] for i in real_idx if vals[i] > 0][:n]
        return fake_words, real_words
    except Exception:
        return [], []


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<html><body><p>Static UI not found. Use <a href='/docs'>/docs</a></p></body></html>")


@app.get("/health")
async def health():
    try:
        get_pipeline()
        return {"status": "ok", "model_loaded": True, "version": "1.0.0"}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    pipeline = get_pipeline()
    text = req.text.strip()
    proba = pipeline.predict_proba([text])[0]
    real_prob, fake_prob = float(proba[0]), float(proba[1])
    label = "FAKE" if fake_prob > real_prob else "REAL"
    confidence = max(real_prob, fake_prob)
    fake_words, real_words = get_top_keywords(pipeline, text)
    warning = None
    if confidence < 0.65:
        warning = "Low confidence — model is uncertain. Verify with multiple sources."
    elif confidence < 0.75:
        warning = "Moderate confidence — consider independent verification."
    return PredictResponse(
        label=label, confidence=round(confidence, 4),
        real_probability=round(real_prob, 4), fake_probability=round(fake_prob, 4),
        top_fake_indicators=fake_words, top_real_indicators=real_words,
        word_count=len(text.split()), warning=warning,
    )


@app.get("/examples")
async def examples():
    return {
        "real": {"text": "The Federal Reserve raised interest rates by 25 basis points on Wednesday, the tenth consecutive increase as the central bank continues its fight against inflation."},
        "fake": {"text": "SHOCKING: Government secretly embedding microchips in COVID vaccines to track and control citizens worldwide — leaked documents PROVE the deep state conspiracy mainstream media refuses to cover!"}
    }
