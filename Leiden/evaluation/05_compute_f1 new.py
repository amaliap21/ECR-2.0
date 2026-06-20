"""
Step 5: Compute accuracy + per-class F1 + macro-F1 for the 5 models on the gold
samples. Per topic AND overall. Save CSV + bar-chart visualization.

Reads the merged 6-topic eval (outputs/summary/eval_all6.csv) when present —
that file is produced by eval5k_assemble.py and combines the 3 original topics
(60 error-samples each) with the 3 new topics (time-based 5k gold). If it is
absent, it falls back to the original 180-sample file so old runs still work.

Topics (evaluation splits) are derived from the data, so adding topics needs no
code change. Rows with a missing gold label, and per-model cells with a missing
prediction, are skipped automatically.

Best model = highest overall macro-F1.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, classification_report)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs", "summary")
IN_PATH = os.path.join(HERE, "outputs", "eval5k", "combined_normalized.csv")

MODELS = {
    "Fine-tuned (XLM-RoBERTa)":  "finetuned_label",
    "RoBERTa API (cached)":      "roberta_api_cached_label",
    "IndoBERT API (No Limit)":   "roberta_api_label",
    "GPT-5.4 mini":              "gpt_label",
    "IndoBERT":                  "indobert_label",
}
LABELS = ["negative", "neutral", "positive"]


def norm(s):
    if not isinstance(s, str):
        return "neutral"
    s = s.strip().lower()
    return {"positif": "positive", "negatif": "negative", "netral": "neutral"}.get(s, s)


def score(y_true, y_pred):
    # Drop pairs where the model prediction is missing (NaN/empty) so a model
    # that hasn't finished inference is scored only on the rows it has.
    yt, yp = [], []
    for t, p in zip(y_true, y_pred):
        if isinstance(p, str) and p in LABELS and t in LABELS:
            yt.append(t)
            yp.append(p)
    if not yt:
        return {k: float("nan") for k in
                ("accuracy", "macro_f1", "f1_neg", "f1_neu", "f1_pos",
                 "precision", "recall")} | {"n_scored": 0}
    per_class = f1_score(yt, yp, labels=LABELS, average=None, zero_division=0)
    return {
        "accuracy":  accuracy_score(yt, yp),
        "macro_f1":  f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0),
        "f1_neg":    per_class[0],
        "f1_neu":    per_class[1],
        "f1_pos":    per_class[2],
        "precision": precision_score(yt, yp, labels=LABELS, average="macro", zero_division=0),
        "recall":    recall_score(yt, yp, labels=LABELS, average="macro", zero_division=0),
        "n_scored":  len(yt),
    }


def main():
    print(f"Reading {IN_PATH}")
    df = pd.read_csv(IN_PATH)
    df["manual_label"] = df["manual_label"].map(norm)
    for col in MODELS.values():
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = df[col].map(norm)
    # Only rows that have a gold label count toward the evaluation.
    df = df[df["manual_label"].isin(LABELS)].reset_index(drop=True)

    # Splits = overall + each topic present in the data (sorted, stable).
    topics = sorted(t for t in df["topic"].dropna().unique())
    splits = [("overall", df)] + [(t, df[df["topic"] == t]) for t in topics]

    rows = []
    for split_name, sub in splits:
        y_true = sub["manual_label"].tolist()
        for name, col in MODELS.items():
            s = score(y_true, sub[col].tolist())
            s["split"] = split_name
            s["model"] = name
            s["n"] = len(sub)
            rows.append(s)

    res = pd.DataFrame(rows)[["split", "model", "n", "n_scored", "accuracy",
                               "macro_f1", "f1_neg", "f1_neu", "f1_pos",
                               "precision", "recall"]]
    csv_path = os.path.join(OUT_DIR, "model_comparison.csv")
    res.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")
    print(res.to_string(index=False))

    # --- Best model on overall macro_f1 ---
    overall = res[res["split"] == "overall"].sort_values("macro_f1", ascending=False)
    best = overall.iloc[0]["model"]
    print(f"\nBest model (overall macro-F1): {best} "
          f"(macro_f1={overall.iloc[0]['macro_f1']:.4f}, acc={overall.iloc[0]['accuracy']:.4f})")
    with open(os.path.join(OUT_DIR, "best_model.txt"), "w") as f:
        f.write(f"Best model by overall macro-F1: {best}\n")
        f.write(overall.to_string(index=False))

    # --- Visualization: grouped bar chart by topic + overall ---
    split_names = ["overall"] + topics
    model_names = list(MODELS.keys())

    fig, axes = plt.subplots(1, 2, figsize=(4 + 2.4 * len(split_names), 6))
    width = 0.8 / len(model_names)
    x = np.arange(len(split_names))
    colors = ["#2E86AB", "#A23B72", "#6A7FA8", "#F18F01", "#C73E1D"]

    for metric, ax, title in [("macro_f1", axes[0], "Macro-F1"),
                               ("accuracy", axes[1], "Accuracy")]:
        for i, m in enumerate(model_names):
            vals = [res[(res["split"] == sp) & (res["model"] == m)][metric].iloc[0]
                    for sp in split_names]
            ax.bar(x + i * width, vals, width, label=m, color=colors[i % len(colors)])
            for j, v in enumerate(vals):
                if pd.notna(v):
                    ax.text(x[j] + i * width, v + 0.01, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(split_names, rotation=15)
        ax.set_ylabel(title)
        ax.set_title(f"{title} per topic + overall (gold-labeled rows)")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    png = os.path.join(OUT_DIR, "model_comparison.png")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    print(f"Wrote {png}")

    # --- Per-topic detailed classification reports ---
    for topic in topics:
        sub = df[df["topic"] == topic]
        lines = [f"=== {topic} (n={len(sub)}) ==="]
        for name, col in MODELS.items():
            m = sub[sub[col].isin(LABELS)]
            lines.append(f"\n--- {name} (n_scored={len(m)}) ---")
            if m.empty:
                lines.append("(no predictions available)")
                continue
            lines.append(classification_report(m["manual_label"], m[col],
                                                 labels=LABELS, zero_division=0))
            cm = confusion_matrix(m["manual_label"], m[col], labels=LABELS)
            lines.append(f"Confusion matrix (rows=true, cols=pred, order={LABELS}):\n{cm}")
        path = os.path.join(HERE, "outputs", topic, "classification_reports.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
