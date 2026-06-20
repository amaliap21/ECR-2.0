"""
Assemble the final 6-topic evaluation table for compute_f1.

  pool.csv (6 topics x 5,000) + gold_labels.csv + pred_*.csv  ->  eval_all6.csv

Only rows you have already gold-labeled are included, so you can run this at any
point (even mid-labeling) and re-run compute_f1 on whatever is done so far. Rows
missing a model prediction are kept (that cell is left blank and compute_f1
skips it per-model). The old 180 error-sample eval is no longer used.

Output columns (exact order):
    cm_cell, cleaned_text, indobert_label, roberta_api_cached_label,
    manual_label, topic, gpt_label, roberta_api_cached_confidence, text,
    indobert_api_label, indobert_api_confidence, finetuned_label,
    finetuned_confidence
"""

import os
import pandas as pd

from eval5k_config import EVAL_DIR, SUMMARY_DIR, ensure_dirs, norm_label

POOL = os.path.join(EVAL_DIR, "pool.csv")
GOLD = os.path.join(EVAL_DIR, "gold_labels.csv")
OUT_ALL = os.path.join(SUMMARY_DIR, "eval_all6.csv")

# Exact output schema (order matters). The "IndoBERT API (No Limit)" model =
# the self-hosted RoBERTa endpoint, surfaced here as indobert_api_label/_confidence.
FINAL_COLS = ["cm_cell", "cleaned_text", "indobert_label",
              "roberta_api_cached_label", "manual_label", "topic", "gpt_label",
              "roberta_api_cached_confidence", "text", "indobert_api_label",
              "indobert_api_confidence", "finetuned_label", "finetuned_confidence"]

# pred cache file key -> (source col in cache, output col in FINAL_COLS)
PRED_CACHES = {
    "finetuned":   [("finetuned_label", "finetuned_label"),
                    ("finetuned_confidence", "finetuned_confidence")],
    "indobert":    [("indobert_label", "indobert_label")],
    "roberta_api": [("roberta_api_label", "indobert_api_label"),
                    ("roberta_api_confidence", "indobert_api_confidence")],
    "gpt":         [("gpt_label", "gpt_label")],
}


def build():
    pool = pd.read_csv(POOL)
    if not os.path.exists(GOLD):
        print("No gold_labels.csv yet — run eval5k_label.py to enter gold labels.")
        return pd.DataFrame(columns=FINAL_COLS)

    gold = pd.read_csv(GOLD)[["uid", "manual_label"]]
    gold["manual_label"] = gold["manual_label"].map(norm_label)
    df = pool.merge(gold, on="uid", how="inner")  # only gold-labeled rows
    print(f"{len(df):,} gold-labeled rows across {df['topic'].nunique()} topics")

    # roberta_api_cached_label/confidence are already in pool; merge the rest.
    for key, colmap in PRED_CACHES.items():
        p = os.path.join(EVAL_DIR, f"pred_{key}.csv")
        if os.path.exists(p):
            cdf = pd.read_csv(p)
            ren = {src: out for src, out in colmap}
            cdf = cdf[["uid"] + [src for src, _ in colmap]].rename(columns=ren)
            for _, out in colmap:
                if out.endswith("_label"):
                    cdf[out] = cdf[out].map(norm_label)
            df = df.merge(cdf, on="uid", how="left")
            lab = colmap[0][1]
            miss = df[lab].isna().sum()
            print(f"  {lab:28s}: {len(df) - miss:,}/{len(df):,} present"
                  + (f"  ({miss} missing)" if miss else ""))
        else:
            for _, out in colmap:
                df[out] = pd.NA
            print(f"  {colmap[0][1]:28s}: cache missing "
                  f"(run eval5k_infer.py --models {key})")

    df["roberta_api_cached_label"] = df["roberta_api_cached_label"].map(norm_label)
    df["cleaned_text"] = df["text"]
    df["cm_cell"] = pd.NA  # time-based sample; no confusion-matrix cell
    return df.reindex(columns=FINAL_COLS)


def main():
    ensure_dirs()
    allrows = build()
    allrows.to_csv(OUT_ALL, index=False, encoding="utf-8")
    print(f"\nWrote {OUT_ALL}  ({len(allrows):,} rows)")
    if len(allrows):
        print(allrows.groupby("topic").size().to_string())


if __name__ == "__main__":
    main()
