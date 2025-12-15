import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import sys
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))
from src.multitask_model import MultiTaskDistilBert


class UnifiedDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int = 256, maps=None):
        self.items = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.maps = maps
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                if "text" in ex:
                    self.items.append(ex)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        text = ex["text"]
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        def norm(s: str):
            s = str(s).strip().lower().replace("_", " ").replace("-", " ")
            if s in {"lovelife"}:
                s = "love life"
            if s in {"familyissues"}:
                s = "family issues"
            if s in {"healthissues"}:
                s = "health issues"
            if s in {"moodissues"}:
                s = "mood issues"
            return s
        y_sent = torch.tensor(self.maps["sentiment"][norm(ex["sentiment"])], dtype=torch.long)
        y_tox = torch.tensor(self.maps["toxicity"][norm(ex["toxicity"])], dtype=torch.long)
        y_topic = torch.tensor(self.maps["topic"][norm(ex["topic"])], dtype=torch.long)
        return input_ids, attention_mask, y_sent, y_tox, y_topic


def train():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "datasets" / "unified.jsonl"
    out_dir = root / "models" / "multitask"
    out_dir.mkdir(parents=True, exist_ok=True)
    maps = {
        "sentiment": {"positive": 0, "neutral": 1, "negative": 2},
        "toxicity": {"safe": 0, "offensive/hate speech": 1, "spam": 2},
        "topic": {"career": 0, "love life": 1, "family issues": 2, "health issues": 3, "mood issues": 4},
    }
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    ds = UnifiedDataset(data_path, tokenizer, maps=maps)
    val_size = max(int(0.1 * len(ds)), 1)
    train_size = len(ds) - val_size
    ds_train, ds_val = random_split(ds, [train_size, val_size])
    train_loader = DataLoader(ds_train, batch_size=16, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=32, shuffle=False)
    model = MultiTaskDistilBert()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = AdamW(model.parameters(), lr=2e-5)
    total_steps = max(len(train_loader) * 4, 1)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=max(total_steps // 10, 1), num_training_steps=total_steps)
    # class weights
    def compute_weights(ds_all, key):
        counts = [0] * len(maps[key])
        for ex in ds_all.items:
            def norm(s: str):
                s = str(s).strip().lower().replace("_", " ").replace("-", " ")
                if s in {"lovelife"}:
                    s = "love life"
                if s in {"familyissues"}:
                    s = "family issues"
                if s in {"healthissues"}:
                    s = "health issues"
                if s in {"moodissues"}:
                    s = "mood issues"
                return s
            counts[maps[key][norm(ex[key])]] += 1
        total = sum(counts)
        weights = [total / (c if c > 0 else 1) for c in counts]
        s = sum(weights)
        weights = [w / s for w in weights]
        return torch.tensor(weights, dtype=torch.float32).to(device)
    w_sent = compute_weights(ds, "sentiment")
    w_tox = compute_weights(ds, "toxicity")
    w_topic = compute_weights(ds, "topic")
    loss_sent = nn.CrossEntropyLoss(weight=w_sent)
    loss_tox = nn.CrossEntropyLoss(weight=w_tox)
    loss_topic = nn.CrossEntropyLoss(weight=w_topic)
    best_val = float("inf")
    best_epoch = -1
    for epoch in range(4):
        model.train()
        for input_ids, attention_mask, y_sent, y_tox, y_topic in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            y_sent = y_sent.to(device)
            y_tox = y_tox.to(device)
            y_topic = y_topic.to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_sent(out["sentiment"], y_sent) + loss_tox(out["toxicity"], y_tox) + loss_topic(out["topic"], y_topic)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, attention_mask, y_sent, y_tox, y_topic in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                y_sent = y_sent.to(device)
                y_tox = y_tox.to(device)
                y_topic = y_topic.to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                vloss = loss_sent(out["sentiment"], y_sent) + loss_tox(out["toxicity"], y_tox) + loss_topic(out["topic"], y_topic)
                val_loss += float(vloss.item())
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            model.encoder.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            torch.save(model.sentiment_head.state_dict(), out_dir / "sentiment_head.bin")
            torch.save(model.toxicity_head.state_dict(), out_dir / "toxicity_head.bin")
            torch.save(model.topic_head.state_dict(), out_dir / "topic_head.bin")
    with (out_dir / "label_maps.json").open("w", encoding="utf-8") as f:
        json.dump(maps, f)


if __name__ == "__main__":
    train()
