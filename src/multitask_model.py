from typing import Dict
import torch
from torch import nn
from transformers import AutoModel


class MultiTaskDistilBert(nn.Module):
    def __init__(self, base_model_name: str = "distilbert-base-multilingual-cased", n_sentiment: int = 3, n_toxicity: int = 3, n_topic: int = 5):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden = self.encoder.config.hidden_size
        self.sentiment_head = nn.Linear(hidden, n_sentiment)
        self.toxicity_head = nn.Linear(hidden, n_toxicity)
        self.topic_head = nn.Linear(hidden, n_topic)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state[:, 0]
        logits_sentiment = self.sentiment_head(h)
        logits_toxicity = self.toxicity_head(h)
        logits_topic = self.topic_head(h)
        return {
            "sentiment": logits_sentiment,
            "toxicity": logits_toxicity,
            "topic": logits_topic,
        }
