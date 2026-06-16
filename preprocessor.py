"""
Standalone preprocessor module.
Must be importable by both train.py and the FastAPI app.
"""
import re

def clean_text(text: str) -> str:
    """TF-IDF preprocessor: lowercase, remove URLs/HTML/punct."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
