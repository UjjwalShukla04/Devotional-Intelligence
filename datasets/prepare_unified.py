import argparse
import json
from pathlib import Path
import pandas as pd


def from_devotional(csv_path: Path, limit: int | None):
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "text": str(r["text"]),
                "sentiment": "neutral",
                "toxicity": "safe",
                "topic": str(r["topic"]).lower(),
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def from_moderation(csv_path: Path, limit: int | None):
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "text": str(r["text"]),
                "sentiment": str(r.get("sentiment", "neutral")).lower(),
                "toxicity": str(r["toxicity"]).lower(),
                "topic": "mood issues",
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def try_from_sst2(limit: int | None):
    try:
        from datasets import load_dataset
        ds = load_dataset("glue", "sst2")
        rows = []
        for ex in ds["train"]:
            sentiment = "positive" if int(ex["label"]) == 1 else "negative"
            rows.append(
                {
                    "text": str(ex["sentence"]),
                    "sentiment": sentiment,
                    "toxicity": "safe",
                    "topic": "mood issues",
                }
            )
            if limit and len(rows) >= limit:
                break
        return rows
    except Exception:
        return []


def try_from_jigsaw(limit: int | None):
    try:
        from datasets import load_dataset
        ds = load_dataset("civil_comments")
        rows = []
        for ex in ds["train"]:
            tox_score = float(ex.get("toxicity", 0.0))
            label = "offensive/hate speech" if tox_score >= 0.5 else "safe"
            text = str(ex.get("text", ""))
            if any(x in text.lower() for x in ["http", "www.", "click", "buy now", "limited offer"]):
                label = "spam"
            rows.append(
                {
                    "text": text,
                    "sentiment": "neutral",
                    "toxicity": label,
                    "topic": "mood issues",
                }
            )
            if limit and len(rows) >= limit:
                break
        return rows
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-sst2", action="store_true")
    parser.add_argument("--include-jigsaw", action="store_true")
    parser.add_argument("--include-devotional", action="store_true")
    parser.add_argument("--include-moderation", action="store_true")
    parser.add_argument("--augment-hinglish", action="store_true")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--out", type=str, default="datasets/unified.jsonl")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.include_devotional:
        rows.extend(from_devotional(root / "datasets" / "devotional_topics.csv", args.limit))
    if args.include_moderation:
        rows.extend(from_moderation(root / "datasets" / "moderation.csv", args.limit))
    if args.include_sst2:
        rows.extend(try_from_sst2(args.limit))
    if args.include_jigsaw:
        rows.extend(try_from_jigsaw(args.limit))
    if args.augment_hinglish:
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
        aug = []
        for r in rows:
            t = r["text"]
            tl = t.lower()
            for en, hi in lex.items():
                tl = tl.replace(en, hi)
            aug.append({"text": tl, "sentiment": r["sentiment"], "toxicity": r["toxicity"], "topic": r["topic"]})
        rows.extend(aug)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
