import json
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


class MultiTaskInference:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.encoder = AutoModel.from_pretrained(self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, fix_mistral_regex=True)
        with (self.model_dir / "label_maps.json").open("r", encoding="utf-8") as f:
            self.maps = json.load(f)
        self.sentiment_labels = ["Positive", "Neutral", "Negative"]
        self.toxicity_labels = ["Safe", "Offensive/Hate Speech", "Spam"]
        self.topic_labels = ["Career", "Love Life", "Family Issues", "Health Issues", "Mood Issues"]
        self.sentiment_head = torch.nn.Linear(self.encoder.config.hidden_size, len(self.sentiment_labels))
        self.toxicity_head = torch.nn.Linear(self.encoder.config.hidden_size, len(self.toxicity_labels))
        self.topic_head = torch.nn.Linear(self.encoder.config.hidden_size, len(self.topic_labels))
        self.sentiment_head.load_state_dict(torch.load(self.model_dir / "sentiment_head.bin", map_location="cpu"))
        self.toxicity_head.load_state_dict(torch.load(self.model_dir / "toxicity_head.bin", map_location="cpu"))
        self.topic_head.load_state_dict(torch.load(self.model_dir / "topic_head.bin", map_location="cpu"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.sentiment_head.to(self.device)
        self.toxicity_head.to(self.device)
        self.topic_head.to(self.device)
        self.encoder.eval()
        self.sentiment_head.eval()
        self.toxicity_head.eval()
        self.topic_head.eval()

    def predict(self, text: str):
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            h = out.last_hidden_state[:, 0]
            s = self.sentiment_head(h)
            t = self.toxicity_head(h)
            p = self.topic_head(h)
        s_prob = torch.softmax(s, dim=-1).cpu().numpy()[0]
        t_prob = torch.softmax(t, dim=-1).cpu().numpy()[0]
        p_prob = torch.softmax(p, dim=-1).cpu().numpy()[0]
        s_idx = int(np.argmax(s_prob))
        t_idx = int(np.argmax(t_prob))
        p_idx = int(np.argmax(p_prob))
        return {
            "sentiment": {"label": self.sentiment_labels[s_idx], "confidence": float(s_prob[s_idx])},
            "toxicity": {"label": self.toxicity_labels[t_idx], "confidence": float(t_prob[t_idx])},
            "topic": {"label": self.topic_labels[p_idx], "confidence": float(p_prob[p_idx])},
        }
