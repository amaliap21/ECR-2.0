"""
Step 6: Confidence bucket analysis — MODULAR, per dataset.

The `sentiment_confidence_level` column in total_data_cleaned_*.csv is the
No-Limit RoBERTa API confidence (shipped pre-computed in the IndSight export),
not the fine-tuned model's confidence. We therefore run TWO bucket analyses
per topic:

  (i)  RoBERTa-API confidence buckets  — scalable to the full 600k dataset
  (ii) Fine-tuned confidence buckets   — computed only on the 60-sample gold
       (fine-tuned is never run on the full 600k)

Buckets:  [0, 0.8)  and  [0.8, 1]

Outputs per topic:
  - confidence_buckets.csv / .png            : RoBERTa-API counts (full + gold)
  - gold_f1_by_bucket_roberta.csv / .png     : RoBERTa-cached F1 per bucket
  - gold_f1_by_bucket_finetuned.csv / .png   : Fine-tuned F1 per bucket
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

GOLD_PATH = os.path.join(HERE, "outputs", "summary", "error_gold_180_all4.csv")

TOPIC_TO_SOURCE = {
    "boikot":          os.path.join(ROOT, "total_data_cleaned_mcd.csv"),
    "vaksin":          os.path.join(ROOT, "total_data_cleaned_vaksin.csv"),
    "indonesia_gelap": os.path.join(ROOT, "total_data_cleaned_indonesia_gelap.csv"),
}

LABELS = ["negative", "neutral", "positive"]
BUCKETS = [("[0, 0.8)", 0.0, 0.8, False),  # right-exclusive
           ("[0.8, 1]", 0.8, 1.0, True)]


def norm(s):
    if not isinstance(s, str):
        return "neutral"
    s = s.strip().lower()
    return {"positif": "positive", "negatif": "negative", "netral": "neutral"}.get(s, s)


def bucket_of(v):
    if pd.isna(v):
        return None
    if 0.0 <= v < 0.8:
        return "[0, 0.8)"
    if 0.8 <= v <= 1.0:
        return "[0.8, 1]"
    return None


def analyze_topic(topic: str, gold: pd.DataFrame):
    print(f"\n=== {topic} ===")
    out_dir = os.path.join(HERE, "outputs", topic)
    os.makedirs(out_dir, exist_ok=True)

    # --- FULL dataset counts ---
    src = TOPIC_TO_SOURCE[topic]
    full = pd.read_csv(src, usecols=["sentiment_confidence_level"])
    full["bucket"] = full["sentiment_confidence_level"].map(bucket_of)
    full_counts = full["bucket"].value_counts().reindex([b[0] for b in BUCKETS]).fillna(0).astype(int)
    full_total = int(full_counts.sum())
    print(f"  full dataset rows with confidence: {full_total:,} / {len(full):,}")
    print(f"    counts: {full_counts.to_dict()}")

    # --- 60 gold samples for this topic (bucketed by RoBERTa API confidence) ---
    g = gold[gold["topic"] == topic].copy()
    g["bucket"] = g["roberta_api_cached_confidence"].map(bucket_of)
    gold_counts = g["bucket"].value_counts().reindex([b[0] for b in BUCKETS]).fillna(0).astype(int)
    print(f"  gold rows: {len(g)}  bucket counts (RoBERTa conf): {gold_counts.to_dict()}")

    # --- CSV 1: bucket counts table ---
    tbl = pd.DataFrame({
        "bucket": [b[0] for b in BUCKETS],
        "full_dataset_count":        full_counts.values,
        "full_dataset_percent":      (full_counts.values / max(full_total, 1) * 100).round(2),
        "gold_sample_count":         gold_counts.values,
        "gold_sample_percent":       (gold_counts.values / max(len(g), 1) * 100).round(2),
    })
    tbl_path = os.path.join(out_dir, "confidence_buckets.csv")
    tbl.to_csv(tbl_path, index=False)
    print(f"  wrote {tbl_path}")

    # --- PNG 1: bucket counts bar plot (log-y for full dataset) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(BUCKETS))
    axes[0].bar(x, full_counts.values, color=["#F18F01", "#2E86AB"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([b[0] for b in BUCKETS])
    axes[0].set_title(f"{topic}: full dataset ({full_total:,} rows)")
    axes[0].set_ylabel("rows")
    for i, v in enumerate(full_counts.values):
        axes[0].text(i, v, f"{v:,}\n({v/max(full_total,1)*100:.1f}%)",
                    ha="center", va="bottom", fontsize=10)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].margins(y=0.15)

    axes[1].bar(x, gold_counts.values, color=["#F18F01", "#2E86AB"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([b[0] for b in BUCKETS])
    axes[1].set_title(f"{topic}: gold 60 samples")
    axes[1].set_ylabel("rows")
    for i, v in enumerate(gold_counts.values):
        axes[1].text(i, v, f"{v}\n({v/max(len(g),1)*100:.1f}%)",
                    ha="center", va="bottom", fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].margins(y=0.15)

    plt.suptitle(f"Confidence buckets — {topic}", fontsize=13, y=1.02)
    plt.tight_layout()
    png_path = os.path.join(out_dir, "confidence_buckets.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {png_path}")

    # --- F1 per bucket on gold samples — TWO analyses ---
    g["manual_norm"]          = g["manual_label"].map(norm)
    g["roberta_cached_norm"]  = g["roberta_api_cached_label"].map(norm)
    g["finetuned_norm"]       = g["finetuned_label"].map(norm)

    def f1_by_bucket(conf_col, pred_col, label):
        rows = []
        for name, lo, hi, incl_hi in BUCKETS:
            if incl_hi:
                sub = g[(g[conf_col] >= lo) & (g[conf_col] <= hi)]
            else:
                sub = g[(g[conf_col] >= lo) & (g[conf_col] < hi)]
            if len(sub) == 0:
                rows.append({"bucket": name, "n": 0, "accuracy": None, "macro_f1": None,
                             "f1_neg": None, "f1_neu": None, "f1_pos": None})
                continue
            y, p = sub["manual_norm"].tolist(), sub[pred_col].tolist()
            per = f1_score(y, p, labels=LABELS, average=None, zero_division=0)
            rows.append({
                "bucket": name, "n": len(sub),
                "accuracy": round(accuracy_score(y, p), 4),
                "macro_f1": round(f1_score(y, p, labels=LABELS, average="macro", zero_division=0), 4),
                "f1_neg": round(per[0], 4), "f1_neu": round(per[1], 4), "f1_pos": round(per[2], 4),
            })
        return pd.DataFrame(rows)

    def plot_f1(tbl, title, path):
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(BUCKETS))
        macro = [r or 0 for r in tbl["macro_f1"].tolist()]
        acc   = [r or 0 for r in tbl["accuracy"].tolist()]
        w = 0.35
        ax.bar(x - w/2, macro, w, label="Macro-F1", color="#2E86AB")
        ax.bar(x + w/2, acc,   w, label="Accuracy", color="#F18F01")
        for i, (m, a, n) in enumerate(zip(macro, acc, tbl["n"].tolist())):
            ax.text(x[i] - w/2, m + 0.01, f"{m:.2f}", ha="center", fontsize=9)
            ax.text(x[i] + w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)
            ax.text(x[i], -0.08, f"n={n}", ha="center", fontsize=9, color="gray",
                    transform=ax.get_xaxis_transform())
        ax.set_xticks(x); ax.set_xticklabels([b[0] for b in BUCKETS])
        ax.set_ylim(0, 1.1); ax.set_ylabel("score")
        ax.set_title(title)
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  wrote {path}")

    # (a) RoBERTa API cached: bucket by roberta_api_cached_confidence,
    #     evaluate roberta_api_cached_label vs manual
    rob_tbl = f1_by_bucket("roberta_api_cached_confidence", "roberta_cached_norm", "roberta_cached")
    rob_path = os.path.join(out_dir, "gold_f1_by_bucket_roberta.csv")
    rob_tbl.to_csv(rob_path, index=False); print(f"  wrote {rob_path}"); print(rob_tbl.to_string(index=False))
    plot_f1(rob_tbl, f"RoBERTa API (cached) F1 per conf bucket — {topic}",
            os.path.join(out_dir, "gold_f1_by_bucket_roberta.png"))

    # (b) Fine-tuned: bucket by finetuned_confidence, evaluate finetuned_label vs manual
    ft_tbl = f1_by_bucket("finetuned_confidence", "finetuned_norm", "finetuned")
    ft_path = os.path.join(out_dir, "gold_f1_by_bucket_finetuned.csv")
    ft_tbl.to_csv(ft_path, index=False); print(f"  wrote {ft_path}"); print(ft_tbl.to_string(index=False))
    plot_f1(ft_tbl, f"Fine-tuned (XLM-RoBERTa) F1 per conf bucket — {topic}",
            os.path.join(out_dir, "gold_f1_by_bucket_finetuned.png"))

    tbl["topic"] = topic
    rob_tbl["topic"] = topic; rob_tbl["model"] = "RoBERTa API (cached)"
    ft_tbl["topic"]  = topic; ft_tbl["model"]  = "Fine-tuned (XLM-RoBERTa)"
    return tbl, pd.concat([rob_tbl, ft_tbl], ignore_index=True)


def main():
    gold = pd.read_csv(GOLD_PATH)
    all_counts, all_f1 = [], []
    for topic in TOPIC_TO_SOURCE:
        t, f = analyze_topic(topic, gold)
        all_counts.append(t); all_f1.append(f)

    summary_dir = os.path.join(HERE, "outputs", "summary")
    os.makedirs(summary_dir, exist_ok=True)
    pd.concat(all_counts, ignore_index=True).to_csv(
        os.path.join(summary_dir, "confidence_buckets_all_topics.csv"), index=False)
    pd.concat(all_f1, ignore_index=True).to_csv(
        os.path.join(summary_dir, "gold_f1_by_bucket_all_topics.csv"), index=False)
    print(f"\nWrote summary CSVs in {summary_dir}")


if __name__ == "__main__":
    main()
