"""
train.py — Fake News Detector: Model Training Script
======================================================
Uses TF-IDF vectorization + Logistic Regression (fast, explainable, strong baseline).
Swap the dataset loading section for the real Kaggle dataset when you have it.

Dataset options (use one):
  1. Kaggle "Fake and Real News Dataset" by Clément Bisaillon
     → https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
     → Download: Fake.csv + True.csv
  2. LIAR dataset (more nuanced, 6-class)
     → https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

Run:
  python train.py                  # uses built-in sample data
  python train.py --real-data      # uses Fake.csv + True.csv from ./data/
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
import re

# ─── Text Preprocessing ─────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Basic NLP preprocessing pipeline."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)       # remove URLs
    text = re.sub(r"<.*?>", " ", text)                 # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)              # keep only letters
    text = re.sub(r"\s+", " ", text).strip()           # collapse whitespace
    return text


# ─── Sample Data (used when --real-data not passed) ─────────────────────────

SAMPLE_DATA = {
    "text": [
        # REAL news style
        "The Federal Reserve raised interest rates by 25 basis points on Wednesday, marking the tenth consecutive increase as the central bank continues its fight against inflation.",
        "Scientists at CERN announced new findings related to the Higgs boson particle, providing deeper insight into the fundamental structure of matter.",
        "The United Nations Security Council met to discuss humanitarian aid access in conflict zones, with member nations debating resolution terms.",
        "Apple reported quarterly earnings of 94.8 billion dollars, exceeding analyst expectations by approximately 2 billion dollars.",
        "NASA's Perseverance rover has collected its 20th rock sample on Mars, part of a cache planned for return to Earth in the early 2030s.",
        "The European Central Bank has maintained its benchmark interest rate at current levels, citing ongoing uncertainty in global energy markets.",
        "Health officials confirmed that new COVID-19 variants are being monitored, while hospitalisation rates remain stable across most regions.",
        "The World Health Organization released updated guidance on antibiotic resistance, urging governments to restrict unnecessary prescriptions.",
        "Stock markets closed higher on Friday after positive jobs data showed the unemployment rate dropped to 3.7 percent.",
        "A new climate agreement was signed by 45 countries pledging to reduce carbon emissions by 40 percent before 2035.",
        "Research published in the journal Nature identified a genetic marker associated with increased risk of certain autoimmune diseases.",
        "The International Monetary Fund revised its global growth forecast to 3.1 percent for the current fiscal year.",
        # FAKE news style
        "BREAKING: Government secretly putting microchips in COVID vaccines to track citizens, leaked documents PROVE it!",
        "Scientists CONFIRM: 5G towers cause cancer and the mainstream media is HIDING the truth from you!",
        "SHOCKING: Politician caught on tape admitting to election fraud, mainstream media refuses to cover it!",
        "Deep state operatives have been caught trying to poison the water supply, whistleblower reveals the TRUTH!",
        "BOMBSHELL: Global elites planning to replace cash with digital currency to control your every purchase!",
        "Doctors BANNED from telling you this simple cure for cancer that Big Pharma doesn't want you to know!",
        "EXPOSED: The moon landing was faked in a Hollywood studio, newly released NASA files CONFIRM it!",
        "URGENT: New world order plans revealed — they want to reduce world population by 90 percent by 2030!",
        "Chemtrails confirmed: Government planes spraying mind-control chemicals over major cities every night!",
        "LEAKED: Top virologist admits COVID-19 was engineered in a lab and released intentionally for profit!",
        "ALERT: Fluoride in tap water is causing mass brain damage, study the government is desperate to suppress!",
        "EXCLUSIVE: Celebrity cabal running child trafficking network from pizza restaurant basement, sources say!",
    ],
    "label": [0]*12 + [1]*12   # 0 = REAL, 1 = FAKE
}


# ─── Training Pipeline ───────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    TF-IDF + Logistic Regression pipeline.
    TF-IDF params tuned for news text:
      - sublinear_tf: dampens very frequent terms
      - ngram_range (1,2): catches bigrams like "breaking news", "fake claim"
      - max_features: keeps top 50k tokens (balanced speed vs accuracy)
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            ngram_range=(1, 2),
            max_features=50_000,
            sublinear_tf=True,
            min_df=2,
            stop_words="english",
        )),
        ("clf", LogisticRegression(
            C=5.0,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
        )),
    ])


def load_kaggle_data(data_dir: str) -> pd.DataFrame:
    """Load Fake.csv + True.csv from Kaggle dataset."""
    fake_path = os.path.join(data_dir, "Fake.csv")
    true_path = os.path.join(data_dir, "True.csv")

    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        raise FileNotFoundError(
            f"Fake.csv or True.csv not found in {data_dir}.\n"
            "Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset"
        )

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 1   # 1 = FAKE
    true_df["label"] = 0   # 0 = REAL

    # Combine title + text for richer features
    for df in [fake_df, true_df]:
        df["text"] = df.get("title", "").fillna("") + " " + df.get("text", "").fillna("")

    combined = pd.concat([fake_df[["text", "label"]], true_df[["text", "label"]]])
    combined = combined.dropna(subset=["text"]).reset_index(drop=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return combined


def train(use_real_data: bool = False, data_dir: str = "./data") -> None:
    print("\n" + "="*55)
    print("  FAKE NEWS DETECTOR — Model Training")
    print("="*55)

    # ── Load data ──
    if use_real_data:
        print(f"\n[1/4] Loading Kaggle dataset from {data_dir}/ ...")
        df = load_kaggle_data(data_dir)
        print(f"      Loaded {len(df):,} articles  |  Fake: {df['label'].sum():,}  |  Real: {(df['label']==0).sum():,}")
    else:
        print("\n[1/4] Using built-in sample data (26 examples)")
        print("      → For production accuracy, use the Kaggle dataset (see --real-data flag)")
        df = pd.DataFrame(SAMPLE_DATA)

    X = df["text"].astype(str)
    y = df["label"]

    # ── Split ──
    test_size = 0.2 if len(df) >= 20 else 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"\n[2/4] Split → Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Train ──
    print("\n[3/4] Training TF-IDF + Logistic Regression pipeline ...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("      Training complete.")

    # ── Evaluate ──
    print("\n[4/4] Evaluation on test set:")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n      Accuracy : {acc:.1%}")
    print("\n" + classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"      Confusion Matrix:\n      {cm}")

    # ── Save model ──
    os.makedirs("./artifacts", exist_ok=True)
    model_path = "./artifacts/model_pipeline.pkl"
    joblib.dump(pipeline, model_path)
    print(f"\n✅  Model saved → {model_path}")
    print("    Run: uvicorn app.main:app --reload  to start the API\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fake news detector")
    parser.add_argument("--real-data", action="store_true",
                        help="Use Kaggle Fake.csv + True.csv from ./data/")
    parser.add_argument("--data-dir", default="./data",
                        help="Directory containing Fake.csv and True.csv")
    args = parser.parse_args()
    train(use_real_data=args.real_data, data_dir=args.data_dir)
