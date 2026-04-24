"""
Step 10: build the 180-row evaluation set from the three
error_samples_labeled.csv files in Leiden/hasil_*/ (60 rows/topic, sampled
from the off-diagonal cells of the RoBERTa-API-vs-IndoBERT/GPT confusion matrix).

The `final_sentiment` column shipped in the IndSight CSV export is the
No-Limit RoBERTa API output (pre-computed upstream). We rename it to
`roberta_api_cached_label` so it's clearly identified as a cached RoBERTa
result, not "fine-tuned". Likewise `sentiment_confidence_level` is
RoBERTa API's confidence; renamed to `roberta_api_cached_confidence`.

Enriched from the per-topic checkpoint (code_<topic>/<topic>_checkpoint_gpt54_sentiment.csv)
by matching on cleaned_text to fill in indobert_label / gpt_label / confidence.

Output: outputs/summary/error_gold_180_with_checkpoint.csv
"""

import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROJECT = os.path.dirname(ROOT)

ERROR_FILES = {
    "boikot":          os.path.join(ROOT, "hasil_boikot",
                                     "boikot_error_samples_labeled.csv"),
    "vaksin":          os.path.join(ROOT, "hasil_vaksin",
                                     "vaksin_error_samples_labeled.csv"),
    "indonesia_gelap": os.path.join(ROOT, "hasil_indonesia_gelap",
                                     "indonesia_gelap_error_samples_labeled.csv"),
}

CHECKPOINTS = {
    "boikot":          os.path.join(ROOT, "code_boikot", "boikot_checkpoint_gpt54_sentiment.csv"),
    "vaksin":          os.path.join(ROOT, "code_vaksin", "vaksin_checkpoint_gpt54_sentiment.csv"),
    "indonesia_gelap": os.path.join(ROOT, "code_indonesia_gelap", "indonesia_gelap_checkpoint_gpt54_sentiment.csv"),
}

OUT_PATH = os.path.join(HERE, "outputs", "summary", "error_gold_180_with_checkpoint.csv")


def norm_text(s):
    return " ".join(str(s).split()).strip() if isinstance(s, str) else ""


def main():
    frames = []
    for topic, path in ERROR_FILES.items():
        df = pd.read_csv(path)
        df["topic"] = topic
        df["_key"] = df["cleaned_text"].map(norm_text)

        cp = pd.read_csv(CHECKPOINTS[topic])
        cp["_key"] = cp["cleaned_text"].map(norm_text)
        keep = ["_key", "indobert_label", "gpt_label", "sentiment_confidence_level"]
        cp = cp[[c for c in keep if c in cp.columns]].drop_duplicates(subset="_key")

        # prefix checkpoint cols to avoid clashes, then resolve
        cp = cp.rename(columns={
            "indobert_label": "_cp_indobert_label",
            "gpt_label":      "_cp_gpt_label",
            "sentiment_confidence_level": "_cp_confidence",
        })
        merged = df.merge(cp, on="_key", how="left")

        # fill indobert_label
        if "indobert_label" not in merged.columns:
            merged["indobert_label"] = None
        merged["indobert_label"] = merged["indobert_label"].fillna(merged.get("_cp_indobert_label"))

        # fill gpt_label
        if "gpt_label" not in merged.columns:
            merged["gpt_label"] = None
        merged["gpt_label"] = merged["gpt_label"].fillna(merged.get("_cp_gpt_label"))

        merged["sentiment_confidence_level"] = merged.get("_cp_confidence")

        drop_cols = [c for c in ["_key", "_cp_indobert_label", "_cp_gpt_label", "_cp_confidence"]
                      if c in merged.columns]
        merged = merged.drop(columns=drop_cols)

        miss_ib   = merged["indobert_label"].isna().sum()
        miss_gpt  = merged["gpt_label"].isna().sum()
        miss_conf = merged["sentiment_confidence_level"].isna().sum()
        print(f"{topic}: rows={len(merged)}  miss indobert={miss_ib}  miss gpt={miss_gpt}  miss conf={miss_conf}")
        frames.append(merged)

    out = pd.concat(frames, ignore_index=True)
    # final_sentiment + sentiment_confidence_level are the No-Limit RoBERTa API
    # output shipped pre-computed inside the IndSight CSV. Rename so it's
    # unambiguous that this is a *cached* RoBERTa result, not fine-tuned.
    out = out.rename(columns={"final_sentiment": "roberta_api_cached_label",
                               "sentiment_confidence_level": "roberta_api_cached_confidence"})
    out["text"] = out["cleaned_text"]
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} ({len(out)} rows)")
    print(f"Cols: {out.columns.tolist()}")


if __name__ == "__main__":
    main()
