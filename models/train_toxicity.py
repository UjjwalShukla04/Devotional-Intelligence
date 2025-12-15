from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import re


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "datasets" / "moderation.csv"
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    def augment_hinglish(text: str) -> str:
        t = str(text).lower()
        lex = {
            "job": "naukri",
            "work": "kaam",
            "career": "career",
            "interview": "interview",
            "promotion": "tarakki",
            "family": "parivar",
            "home": "ghar",
            "health": "sehat",
            "exam": "pariksha",
            "love": "pyaar",
            "relationship": "rishta",
            "stress": "tanaav",
            "worry": "chinta",
            "god": "bhagwan",
            "krishna": "krishna ji",
        }
        for en, hi in lex.items():
            t = t.replace(en, hi)
        t = re.sub(r"\s+", " ", t).strip()
        return t
    x_en = df["text"].astype(str)
    y = df["toxicity"].astype(str).str.lower()
    x_hi = x_en.apply(augment_hinglish)
    x = pd.concat([x_en, x_hi], ignore_index=True)
    y = pd.concat([y, y], ignore_index=True)
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = v.fit_transform(x)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    joblib.dump(clf, models_dir / "toxicity_model.joblib")
    joblib.dump(v, models_dir / "toxicity_vectorizer.joblib")


if __name__ == "__main__":
    main()
