"""
Step 5: Compute accuracy + per-class F1 + macro-F1 for the 4 models on the 210
gold samples. Per topic AND overall. Save CSV + bar-chart visualization.

Best model = highest overall macro-F1.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, classification_report)

HERE = os.path.dirname(os.path.abspath(__file__))
IN_PATH = os.path.join(HERE, "outputs", "summary", "error_gold_180_all4.csv")
OUT_DIR = os.path.join(HERE, "outputs", "summary")

MODELS = {
    "Fine-tuned (XLM-RoBERTa)":  "finetuned_label",
    "IndoBERT API (No Limit)":    "roberta_api_label",
    "RoBERTa API (cached)":      "roberta_api_cached_label",
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
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "macro_f1":  f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "f1_neg":    f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)[0],
        "f1_neu":    f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)[1],
        "f1_pos":    f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)[2],
        "precision": precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
    }


def main():
    df = pd.read_csv(IN_PATH)
    df["manual_label"] = df["manual_label"].map(norm)
    for col in MODELS.values():
        df[col] = df[col].map(norm)

    rows = []
    for split_name, sub in [("overall", df),
                             ("boikot",          df[df["topic"] == "boikot"]),
                             ("vaksin",          df[df["topic"] == "vaksin"]),
                             ("indonesia_gelap", df[df["topic"] == "indonesia_gelap"])]:
        y_true = sub["manual_label"].tolist()
        for name, col in MODELS.items():
            s = score(y_true, sub[col].tolist())
            s["split"] = split_name
            s["model"] = name
            s["n"] = len(sub)
            rows.append(s)

    res = pd.DataFrame(rows)[["split", "model", "n", "accuracy", "macro_f1",
                               "f1_neg", "f1_neu", "f1_pos", "precision", "recall"]]
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
    splits = ["overall", "boikot", "vaksin", "indonesia_gelap"]
    model_names = list(MODELS.keys())

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    width = 0.16
    x = np.arange(len(splits))
    colors = ["#2E86AB", "#A23B72", "#6A7FA8", "#F18F01", "#C73E1D"]

    for metric, ax, title in [("macro_f1", axes[0], "Macro-F1"),
                               ("accuracy", axes[1], "Accuracy")]:
        for i, m in enumerate(model_names):
            vals = [res[(res["split"] == sp) & (res["model"] == m)][metric].iloc[0]
                    for sp in splits]
            ax.bar(x + i * width, vals, width, label=m, color=colors[i])
            for j, v in enumerate(vals):
                ax.text(x[j] + i * width, v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(splits, rotation=10)
        ax.set_ylabel(title)
        ax.set_title(f"{title} on 180 error-sample gold (60/topic, off-diagonal CM cells)")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    png = os.path.join(OUT_DIR, "model_comparison.png")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    print(f"Wrote {png}")

    # --- Per-topic detailed classification reports ---
    for topic in ["boikot", "vaksin", "indonesia_gelap"]:
        sub = df[df["topic"] == topic]
        lines = [f"=== {topic} (n={len(sub)}) ==="]
        for name, col in MODELS.items():
            lines.append(f"\n--- {name} ---")
            lines.append(classification_report(sub["manual_label"], sub[col],
                                                 labels=LABELS, zero_division=0))
            cm = confusion_matrix(sub["manual_label"], sub[col], labels=LABELS)
            lines.append(f"Confusion matrix (rows=true, cols=pred, order={LABELS}):\n{cm}")
        path = os.path.join(HERE, "outputs", topic, "classification_reports.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
