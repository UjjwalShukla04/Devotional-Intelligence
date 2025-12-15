import os
from pathlib import Path
import numpy as np
from transformers import pipeline, AutoTokenizer
from faster_whisper import WhisperModel
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import re
from src.multitask_infer import MultiTaskInference


class InferencePipeline:
    def __init__(self):
        self.root = Path(__file__).resolve().parents[1]
        self.models_dir = self.root / "models"
        self.datasets_dir = self.root / "datasets"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.asr_enabled = os.getenv("DISABLE_ASR", "0").lower() not in {"1", "true", "yes"}
        self.whisper = None
        if self.asr_enabled:
            m = os.getenv("WHISPER_MODEL", "small")
            try:
                self.whisper = WhisperModel(m, device="cpu", compute_type="int8")
            except Exception:
                self.whisper = None
        self.toxicity_model_path = self.models_dir / "toxicity_model.joblib"
        self.toxicity_vectorizer_path = self.models_dir / "toxicity_vectorizer.joblib"
        self.topic_model_path = self.models_dir / "topic_model.joblib"
        self.topic_vectorizer_path = self.models_dir / "topic_vectorizer.joblib"
        self.filler = {"uh", "uhh", "hmm", "hmmm", "mmm", "erm", "um", "umm", "ah", "aah"}
        self.hinglish_map = {
            "padhai": "studies",
            "parivar": "family",
            "pyaar": "love",
            "dil": "heart",
            "bhagwan": "god",
            "krishna ji": "krishna",
            "dua": "prayer",
            "dukh": "sadness",
            "naukri": "job",
            "kaam": "work",
            "sehat": "health",
            "rishta": "relationship",
            "ghar": "home",
            "pariksha": "exam",
            "chinta": "worry",
        }
        self.topic_keywords = {
            "Career": [
                "job",
                "career",
                "interview",
                "resume",
                "cv",
                "promotion",
                "salary",
                "boss",
                "office",
                "company",
                "work",
                "hiring",
                "internship",
                "layoff",
            ],
            "Love Life": [
                "relationship",
                "love",
                "partner",
                "girlfriend",
                "boyfriend",
                "spouse",
                "wife",
                "husband",
                "marriage",
                "dating",
                "breakup",
                "trust",
            ],
            "Family Issues": [
                "family",
                "home",
                "parents",
                "mother",
                "father",
                "child",
                "children",
                "parenting",
                "in-laws",
                "house",
                "conflict",
                "argue",
            ],
            "Health Issues": [
                "health",
                "pain",
                "doctor",
                "medicine",
                "diet",
                "sleep",
                "back pain",
                "nutrition",
                "exercise",
                "fever",
                "illness",
            ],
            "Mood Issues": [
                "stress",
                "anxious",
                "anxiety",
                "depressed",
                "sad",
                "angry",
                "frustrated",
                "mood",
                "lonely",
                "hopeless",
            ],
        }
        if not self.toxicity_model_path.exists() or not self.toxicity_vectorizer_path.exists():
            self._train_toxicity()
        if not self.topic_model_path.exists() or not self.topic_vectorizer_path.exists():
            self._train_topics()
        self.toxicity_clf = joblib.load(self.toxicity_model_path)
        self.toxicity_vec = joblib.load(self.toxicity_vectorizer_path)
        self.topic_clf = joblib.load(self.topic_model_path)
        self.topic_vec = joblib.load(self.topic_vectorizer_path)
        self.toxicity_classes = list(getattr(self.toxicity_clf, "classes_", []))
        self.topic_classes = list(getattr(self.topic_clf, "classes_", []))
        self.toxicity_labels = ["Safe", "Offensive/Hate Speech", "Spam"]
        self.topic_labels = ["Career", "Love Life", "Family Issues", "Health Issues", "Mood Issues"]
        self.mt = None
        mt_dir = self.models_dir / "multitask"
        disable_mt = os.getenv("DISABLE_MT", "0").lower() in {"1", "true", "yes"}
        if not disable_mt and (mt_dir / "config.json").exists() and (mt_dir / "sentiment_head.bin").exists():
            try:
                self.mt = MultiTaskInference(mt_dir)
            except Exception:
                self.mt = None

    def transcribe(self, audio_path: str) -> str:
        if not self.whisper:
            return ""
        segments, _ = self.whisper.transcribe(audio_path, beam_size=5)
        text = "".join([s.text for s in segments]).strip()
        return text

    def preprocess_text(self, text: str) -> str:
        t = text.lower()
        t = re.sub(r"(uh+|hmm+|mmm+|erm+|um+|ah+)", " ", t)
        for f in self.filler:
            t = re.sub(rf"\b{re.escape(f)}\b", " ", t)
        for src, dst in self.hinglish_map.items():
            t = re.sub(rf"\b{re.escape(src)}\b", dst, t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def topic_heuristic(self, t: str):
        scores = {}
        for label, kws in self.topic_keywords.items():
            s = 0
            for kw in kws:
                if kw in t:
                    s += 1
            scores[label] = s
        label = max(scores, key=scores.get)
        score = scores[label]
        return label, score

    def analyze_text(self, text: str):
        t = self.preprocess_text(text)
        if self.mt is not None:
            mt_preds = self.mt.predict(t)
            tp = self.topic_vec.transform([t])
            tp_proba = self.topic_clf.predict_proba(tp)[0]
            tp_idx = int(np.argmax(tp_proba))
            tfidf_topic_label = self.topic_classes[tp_idx] if self.topic_classes else self.topic_labels[tp_idx]
            tfidf_topic_conf = float(tp_proba[tp_idx])
            use_tfidf_topic = False
            if tfidf_topic_conf > mt_preds["topic"]["confidence"]:
                use_tfidf_topic = True
            if mt_preds["topic"]["label"] == "Mood Issues" and tfidf_topic_label != "Mood Issues" and tfidf_topic_conf >= 0.4:
                use_tfidf_topic = True
            h_label, h_score = self.topic_heuristic(t)
            use_heuristic = False
            if h_score >= 1 and h_label != "Mood Issues":
                use_heuristic = True
            topic_out = mt_preds["topic"]
            if use_tfidf_topic:
                topic_out = {"label": tfidf_topic_label, "confidence": tfidf_topic_conf}
            if use_heuristic:
                topic_out = {"label": h_label, "confidence": max(0.7, topic_out.get("confidence", 0.7))}
            return {
                "sentiment": mt_preds["sentiment"],
                "toxicity": mt_preds["toxicity"],
                "topic": topic_out,
            }
        _ = self.tokenizer(t, return_tensors="pt", truncation=True)
        s = self.sentiment(t)[0]
        if s["label"] == "NEGATIVE" and s["score"] >= 0.6:
            sentiment_label = "Negative"
            sentiment_conf = float(s["score"])
        elif s["label"] == "POSITIVE" and s["score"] >= 0.6:
            sentiment_label = "Positive"
            sentiment_conf = float(s["score"])
        else:
            sentiment_label = "Neutral"
            sentiment_conf = float(1.0 - s["score"])
        tx = self.toxicity_vec.transform([t])
        tx_proba = self.toxicity_clf.predict_proba(tx)[0]
        tx_idx = int(np.argmax(tx_proba))
        toxicity_label = self.toxicity_classes[tx_idx] if self.toxicity_classes else self.toxicity_labels[tx_idx]
        toxicity_conf = float(tx_proba[tx_idx])
        tp = self.topic_vec.transform([t])
        tp_proba = self.topic_clf.predict_proba(tp)[0]
        tp_idx = int(np.argmax(tp_proba))
        topic_label = self.topic_classes[tp_idx] if self.topic_classes else self.topic_labels[tp_idx]
        topic_conf = float(tp_proba[tp_idx])
        h_label, h_score = self.topic_heuristic(t)
        if (topic_label == "Mood Issues" and h_label != "Mood Issues" and h_score >= 1) or (topic_conf < 0.5 and h_score >= 2):
            topic_label = h_label
            topic_conf = max(topic_conf, 0.7)
        return {
            "sentiment": {"label": sentiment_label, "confidence": sentiment_conf},
            "toxicity": {"label": toxicity_label, "confidence": toxicity_conf},
            "topic": {"label": topic_label, "confidence": topic_conf},
        }

    def _train_toxicity(self):
        data_path = self.datasets_dir / "moderation.csv"
        df = pd.read_csv(data_path)
        x = df["text"].astype(str)
        y = df["toxicity"].astype(str)
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = v.fit_transform(x)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        joblib.dump(clf, self.toxicity_model_path)
        joblib.dump(v, self.toxicity_vectorizer_path)

    def _train_topics(self):
        data_path = self.datasets_dir / "devotional_topics.csv"
        df = pd.read_csv(data_path)
        x = df["text"].astype(str)
        y = df["topic"].astype(str)
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = v.fit_transform(x)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        joblib.dump(clf, self.topic_model_path)
        joblib.dump(v, self.topic_vectorizer_path)
