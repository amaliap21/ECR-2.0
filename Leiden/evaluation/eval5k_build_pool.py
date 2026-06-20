"""
Build the time-based evaluation pool: SAMPLES_PER_TOPIC evenly time-spaced rows
per NEW topic (korupsi, ijazah, mbg).

For each topic: parse date_created, drop rows with no parseable date or empty
text, sort ascending in time, then take evenly-spaced rows across the whole
timeline (np.linspace over the sorted index). This guarantees uniform coverage
from the earliest to the latest post.

The cached RoBERTa label (final_sentiment, already per-row in the source file)
is carried along as roberta_api_cached_label so that model needs no inference.

Output: outputs/eval5k/pool.csv
Columns: uid, topic, original_id, date_created, text, roberta_api_cached_label
"""

import numpy as np
import pandas as pd

from eval5k_config import (TOPIC_SOURCES, SAMPLES_PER_TOPIC, DATE_FMT,
                           EVAL_DIR, ensure_dirs, make_uid, norm_label)
import os


def sample_topic(topic, path):
    df = pd.read_csv(path, low_memory=False,
                     usecols=lambda c: c in ("date_created", "original_id",
                                             "cleaned_text", "final_sentiment",
                                             "sentiment_confidence_level"))
    n0 = len(df)
    if "sentiment_confidence_level" not in df.columns:
        df["sentiment_confidence_level"] = pd.NA

    # Clean text + parse time. (force plain python str; source may be arrow-backed)
    df["text"] = df["cleaned_text"].map(lambda s: " ".join(str(s).split()).strip())
    df = df[df["text"] != ""]
    df = df[df["text"].str.lower() != "nan"]
    df["dt"] = pd.to_datetime(df["date_created"], format=DATE_FMT, errors="coerce")
    # Fallback for any odd rows that don't match the exact format.
    miss = df["dt"].isna()
    if miss.any():
        df.loc[miss, "dt"] = pd.to_datetime(df.loc[miss, "date_created"],
                                            dayfirst=True, errors="coerce")
    df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)

    n_valid = len(df)
    k = min(SAMPLES_PER_TOPIC, n_valid)
    # Evenly-spaced unique indices across the sorted timeline.
    idx = np.unique(np.linspace(0, n_valid - 1, k).round().astype(int))
    # linspace can collapse to <k unique indices only if n_valid < k (handled above).
    picked = df.iloc[idx].reset_index(drop=True)

    out = pd.DataFrame({
        "uid": [make_uid(topic, i) for i in range(len(picked))],
        "topic": topic,
        "original_id": picked["original_id"].astype(str).values,
        "date_created": picked["date_created"].values,
        "text": picked["text"].values,
        "roberta_api_cached_label": picked["final_sentiment"].map(norm_label).values,
        "roberta_api_cached_confidence": picked["sentiment_confidence_level"].values,
    })
    print(f"  {topic}: source={n0:,}  valid(dated,nonempty)={n_valid:,}  sampled={len(out):,}"
          f"  span={picked['dt'].min()} -> {picked['dt'].max()}")
    return out


def main():
    ensure_dirs()
    frames = [sample_topic(t, p) for t, p in TOPIC_SOURCES.items()]
    pool = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(EVAL_DIR, "pool.csv")
    pool.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nWrote {out_path}  ({len(pool):,} rows)")
    print(pool.groupby("topic").size().to_string())


if __name__ == "__main__":
    main()
