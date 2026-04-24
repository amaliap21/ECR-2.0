"""
Step 12: run the fine-tuned XLM-RoBERTa model from Leiden/finetuned_sentiment/
on the 180 error-sample texts. Adds the *true* fine-tuned prediction:
  - finetuned_label
  - finetuned_confidence

Reads:  outputs/summary/error_gold_180_all4.csv  (output of step 11)
Writes: same file, with finetuned_label + finetuned_confidence columns added.
"""

import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(ROOT, "finetuned_sentiment")
IN_PATH = os.path.join(HERE, "outputs", "summary", "error_gold_180_all4.csv")
OUT_PATH = IN_PATH

BATCH = 16


def main():
    df = pd.read_csv(IN_PATH)
    texts = df["text"].fillna("").astype(str).tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()

    id2label = model.config.id2label
    print(f"id2label: {id2label}")

    labels, confs = [], []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i + BATCH]
            enc = tok(chunk, padding=True, truncation=True, max_length=256,
                      return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            p, idx = probs.max(dim=-1)
            for j in range(len(chunk)):
                labels.append(id2label[int(idx[j].item())])
                confs.append(float(p[j].item()))
            print(f"  {i + len(chunk)}/{len(texts)}")

    df["finetuned_label"] = labels
    df["finetuned_confidence"] = confs
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")
    print("Label counts:", pd.Series(labels).value_counts().to_dict())


if __name__ == "__main__":
    main()
