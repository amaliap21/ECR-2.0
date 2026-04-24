"""
Step 11: call RoBERTa API on the 180 error-sample texts.
Adds roberta_api_label + roberta_api_confidence.
Output: outputs/summary/error_gold_180_all4.csv
"""

import os
import time
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
IN_PATH  = os.path.join(HERE, "outputs", "summary", "error_gold_180_with_checkpoint.csv")
OUT_PATH = os.path.join(HERE, "outputs", "summary", "error_gold_180_all4.csv")

URL = "http://103.67.43.247:9443/sentiment/infer-sentiment-roberta"
BATCH = 20


def call_batch(texts):
    payload = {"list": [{"text": t} for t in texts]}
    for attempt in range(3):
        try:
            r = requests.post(URL, json=payload, timeout=60)
            r.raise_for_status()
            return r.json()["list"]
        except Exception as e:
            print(f"  retry {attempt+1}: {e}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"RoBERTa API failed on batch of {len(texts)}")


def main():
    df = pd.read_csv(IN_PATH)
    texts = df["text"].fillna("").astype(str).tolist()
    labels, probs = [], []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        res = call_batch(chunk)
        for item in res:
            labels.append(item.get("label"))
            probs.append(item.get("probability"))
        print(f"  {i + len(chunk)}/{len(texts)}")
    df["roberta_api_label"] = labels
    df["roberta_api_confidence"] = probs
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")
    print(df["roberta_api_label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
