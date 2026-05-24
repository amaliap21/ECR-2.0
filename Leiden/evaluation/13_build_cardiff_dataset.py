"""
Step 13: build the DeBERTa-v3 fine-tuning dataset from the 3 total_data_cleaned_*.csv
files, keeping only rows with No-Limit RoBERTa confidence >= 0.8 as silver labels.

Pipeline:
  1. Load all 3 topics, concat.
  2. Filter: sentiment_confidence_level >= 0.8 AND non-empty cleaned_text AND
     label in {negative, neutral, positive}.
  3. Dedupe on cleaned_text (keep first — prevents train/test leakage).
  4. Per-topic stratified split 80/10/10 by label (each topic is split on its
     own, then the splits are concatenated). Keeps every topic represented in
     train/val/test at its native class distribution.
    5. Save to Leiden/dataset_cardiff_xlmroberta/{train,val,test}.csv.

Output columns: text, label, topic
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "dataset_cardiff_xlmroberta")

SOURCES = {
    "boikot":          os.path.join(ROOT, "total_data_cleaned_mcd.csv"),
    "vaksin":          os.path.join(ROOT, "total_data_cleaned_vaksin.csv"),
    "indonesia_gelap": os.path.join(ROOT, "total_data_cleaned_indonesia_gelap.csv"),
    "korupsi": os.path.join(ROOT, "total_data_cleaned_korupsi.csv"),
    "ijazah": os.path.join(ROOT, "total_data_cleaned_ijazah.csv"),
    "mbg": os.path.join(ROOT, "total_data_cleaned_mbg.csv"),
}

VALID = {"negative", "neutral", "positive"}
SEED = 42


def norm_label(s):
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    return {"negatif": "negative", "netral": "neutral", "positif": "positive"}.get(s, s)


def main():
    frames = []
    for topic, path in SOURCES.items():
        df = pd.read_csv(path, usecols=["cleaned_text", "final_sentiment",
                                          "sentiment_confidence_level"])
        df["topic"] = topic
        df["label"] = df["final_sentiment"].map(norm_label)
        df["text"]  = df["cleaned_text"].astype(str).str.strip()

        before = len(df)
        df = df[(df["sentiment_confidence_level"] >= 0.8)
                & (df["text"] != "")
                & (df["label"].isin(VALID))]
        print(f"{topic:<20}  raw={before:>8,}  high-conf+valid={len(df):>8,}")
        frames.append(df[["text", "label", "topic"]])

    all_df = pd.concat(frames, ignore_index=True)
    print(f"\ntotal before dedupe: {len(all_df):,}")

    # Dedupe on text — prevents train/test leakage from RTs/duplicates
    all_df = all_df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    print(f"total after dedupe:  {len(all_df):,}")

    print("\nClass distribution (after dedupe):")
    print(all_df["label"].value_counts())
    print("\nPer-topic distribution:")
    print(pd.crosstab(all_df["topic"], all_df["label"]))

    # Per-topic stratified split: split each topic independently by label,
    # then concat. This guarantees every topic is present in train/val/test
    # at its native label distribution, and topics don't get pooled during
    # splitting (so a rare label in one topic won't be stolen by another).
    train_parts, val_parts, test_parts = [], [], []
    for topic, sub in all_df.groupby("topic"):
        tr, tmp = train_test_split(sub, test_size=0.20, random_state=SEED,
                                     stratify=sub["label"])
        va, te = train_test_split(tmp, test_size=0.50, random_state=SEED,
                                    stratify=tmp["label"])
        print(f"  {topic:<20}  train={len(tr):>7,}  val={len(va):>6,}  test={len(te):>6,}")
        train_parts.append(tr)
        val_parts.append(va)
        test_parts.append(te)

    # Shuffle the concatenated splits so batches mix topics during training.
    train = pd.concat(train_parts, ignore_index=True).sample(
        frac=1.0, random_state=SEED).reset_index(drop=True)
    val   = pd.concat(val_parts,   ignore_index=True).sample(
        frac=1.0, random_state=SEED).reset_index(drop=True)
    test  = pd.concat(test_parts,  ignore_index=True).sample(
        frac=1.0, random_state=SEED).reset_index(drop=True)

    for split_name, d in [("train", train), ("val", val), ("test", test)]:
        print(f"\n{split_name}: {len(d):,} rows")
        print(f"  per-label: {d['label'].value_counts().to_dict()}")
        print(f"  per-topic: {d['topic'].value_counts().to_dict()}")

    os.makedirs(OUT_DIR, exist_ok=True)
    train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(OUT_DIR, "val.csv"),   index=False)
    test.to_csv(os.path.join(OUT_DIR, "test.csv"),  index=False)
    print(f"\nWrote {OUT_DIR}/{{train,val,test}}.csv")


if __name__ == "__main__":
    main()
