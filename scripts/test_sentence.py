import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
from src.pipeline import InferencePipeline

if __name__ == "__main__":
    p = InferencePipeline()
    text = "Parenting advice for a stubborn child."
    out = p.analyze_text(text)
    print(out)
