# Final Evaluation Results

**Gold set:** 180 rows (60 per topic) — 10 rows from each of 6 off-diagonal cells of the per-topic disagreement confusion matrix (RoBERTa-vs-IndoBERT for boikot/indonesia_gelap; RoBERTa-vs-GPT for vaksin), then manually labeled.

**Source files** (match the per-topic `*_matching.ipynb` notebooks):
- `Leiden/hasil_boikot/boikot_error_samples_labeled.csv`
- `Leiden/hasil_vaksin/vaksin_error_samples_labeled.csv`
- `Leiden/hasil_indonesia_gelap/indonesia_gelap_error_samples_labeled.csv`

**Pipeline output:** `outputs/summary/error_gold_180_all4.csv`

> **Important clarification on column semantics**
> `final_sentiment` / `sentiment_confidence_level` shipped inside the IndSight CSV export are the **No-Limit RoBERTa API output, pre-computed upstream**, NOT a fine-tuned model.
> So the pipeline produces 4 **distinct** models (Fine-tuned XLM-RoBERTa, RoBERTa API, GPT-5.4 mini, IndoBERT), and we also keep a 5th column — the cached RoBERTa output from the IndSight export — as a sanity check against the fresh API call.

> This is a deliberately adversarial set (all rows are CM-disagreements), so absolute accuracy is lower than a random sample. The *ranking* shows which model is most robust on hard-case disagreements.

---

## 1. 4-Model Comparison (on 180 error-sample gold)

### Overall

| Rank | Model | Accuracy | Macro-F1 | F1 neg | F1 neu | F1 pos |
|---|---|---|---|---|---|---|
| 1 | **GPT-5.4 mini**           | **0.522** | **0.515** | 0.599 | 0.488 | 0.460 |
| 2 | Fine-tuned (XLM-RoBERTa)    | 0.500 | 0.487 | 0.508 | 0.558 | 0.396 |
| 3 | RoBERTa API (No Limit)      | 0.467 | 0.450 | 0.456 | 0.542 | 0.352 |
| 4 | IndoBERT                    | 0.328 | 0.313 | 0.314 | 0.408 | 0.217 |
| — | RoBERTa API (cached in data)| 0.411 | 0.411 | 0.403 | 0.425 | 0.404 |

**Best model: GPT-5.4 mini** (highest overall macro-F1 and accuracy).
**Second: Fine-tuned XLM-RoBERTa** — clear gap over the remaining models.

### Per topic (macro-F1)

| Model | boikot | vaksin | indonesia_gelap |
|---|---|---|---|
| Fine-tuned (XLM-RoBERTa)   | 0.384 | 0.454 | **0.562** |
| RoBERTa API (No Limit)     | 0.379 | 0.388 | 0.532 |
| **GPT-5.4 mini**           | 0.379 | **0.592** | 0.469 |
| IndoBERT                   | 0.222 | 0.387 | 0.287 |
| RoBERTa API (cached)       | **0.431** | 0.206 | 0.531 |

- **indonesia_gelap**: Fine-tuned XLM-RoBERTa wins.
- **vaksin**: GPT-5.4 mini wins; fine-tuned is second. Big gap between fresh RoBERTa (0.388) and cached RoBERTa (0.206) — the No-Limit model has drifted since the 2021 vaksin data was exported.
- **boikot**: tight cluster (fine-tuned / fresh RoBERTa / GPT all around 0.38). Cached RoBERTa happens to top the macro-F1 on this topic.

### Sanity: `matching.ipynb` numbers reproduce

- Vaksin GPT vs manual = 58.33% (matching.ipynb reported 56.67% — 2-row difference explained by 4 manual-label corrections in the Leiden vaksin file since)
- Vaksin **cached** RoBERTa vs manual = 21.67% (matching.ipynb reported 18.33% — same 4 corrections)
- Boikot IndoBERT vs manual = 25.00% ✓ unchanged

Files:
- `summary/model_comparison.csv`, `summary/model_comparison.png`
- `summary/best_model.txt`
- `<topic>/classification_reports.txt`

---

## 2. Confidence Bucket Counts

Buckets on **No-Limit RoBERTa** `sentiment_confidence_level` (that's what's shipped in the full 600k files; fine-tuned is never run on the 600k so its confidence distribution only exists on the 180 gold).

- **Low**  = `[0, 0.8)`
- **High** = `[0.8, 1]`

### Full datasets (≈1.76M rows total)

| Topic           | Total      | `[0, 0.8)`       | `[0.8, 1]`      |
|---|---|---|---|
| boikot (mcd)     | 515,135   | 370,362 (71.9%) | 144,773 (28.1%) |
| vaksin           | 641,138   | 595,705 (92.9%) |  45,433  (7.1%) |
| indonesia_gelap  | 600,270   | 458,899 (76.4%) | 141,371 (23.6%) |

### 180-sample gold subset (60 per topic), bucketed by RoBERTa confidence

| Topic           | `[0, 0.8)` | `[0.8, 1]` |
|---|---|---|
| boikot           | 53 (88.3%) |  7 (11.7%) |
| vaksin           | 56 (93.3%) |  4  (6.7%) |
| indonesia_gelap  | 43 (71.7%) | 17 (28.3%) |

Files:
- `<topic>/confidence_buckets.csv`, `<topic>/confidence_buckets.png`
- `summary/confidence_buckets_all_topics.csv`

---

## 3. F1 per Confidence Bucket — TWO models

### 3a. RoBERTa API (cached) — bucketed by `sentiment_confidence_level`

| Topic           | Bucket     | n  | Accuracy | Macro-F1 |
|---|---|---|---|---|
| boikot           | `[0, 0.8)` | 53 | 0.434 | 0.393 |
| boikot           | `[0.8, 1]` |  7 | 0.714 | **0.711** |
| vaksin           | `[0, 0.8)` | 56 | 0.196 | 0.173 |
| vaksin           | `[0.8, 1]` |  4 | 0.500 | 0.222 |
| indonesia_gelap  | `[0, 0.8)` | 43 | 0.488 | 0.482 |
| indonesia_gelap  | `[0.8, 1]` | 17 | 0.706 | **0.648** |

- **boikot** and **indonesia_gelap** validate the lecturer's hypothesis: high-confidence F1 ≫ low-confidence F1.
- **vaksin** high-confidence bucket has n=4 only; numbers unreliable (rare bucket — 7% of full dataset too).

### 3b. Fine-tuned XLM-RoBERTa — bucketed by its own confidence

| Topic           | Bucket     | n  | Accuracy | Macro-F1 |
|---|---|---|---|---|
| boikot           | `[0, 0.8)` |  2 | 0.500 | 0.333 |
| boikot           | `[0.8, 1]` | 58 | 0.431 | 0.384 |
| vaksin           | `[0, 0.8)` |  0 | —     | —     |
| vaksin           | `[0.8, 1]` | 60 | 0.500 | 0.454 |
| indonesia_gelap  | `[0, 0.8)` |  1 | 0.000 | 0.000 |
| indonesia_gelap  | `[0.8, 1]` | 59 | 0.576 | 0.572 |

- The fine-tuned model is **saturated at high confidence** (mean 0.988, median 1.0) — almost everything falls in `[0.8, 1]`. The bucket hypothesis can't be tested meaningfully on this model with the current buckets; would need finer-grained thresholds (e.g. 0.95 / 0.99 / 1.0).

Files:
- `<topic>/gold_f1_by_bucket_roberta.csv` / `.png`
- `<topic>/gold_f1_by_bucket_finetuned.csv` / `.png`
- `summary/gold_f1_by_bucket_all_topics.csv`

---

## Pipeline — reproduce from scratch

```bash
python Leiden/evaluation/10_build_error_gold.py      # merge 3 error_samples_labeled files, enrich from checkpoints
python Leiden/evaluation/11_roberta_api_error.py     # call No-Limit RoBERTa API on the 180 texts
python Leiden/evaluation/12_finetuned_inference.py   # run fine-tuned XLM-RoBERTa from Leiden/finetuned_sentiment/
python Leiden/evaluation/05_compute_f1.py            # 4(+1)-model F1 + best model + viz
python Leiden/evaluation/06_confidence_buckets.py    # bucket counts + per-bucket F1 (RoBERTa + fine-tuned)
```

## Folder layout

```
Leiden/evaluation/outputs/
├── boikot/
│   ├── classification_reports.txt
│   ├── confidence_buckets.csv / .png
│   ├── gold_f1_by_bucket_roberta.csv / .png
│   └── gold_f1_by_bucket_finetuned.csv / .png
├── vaksin/             (same structure)
├── indonesia_gelap/    (same structure)
└── summary/
    ├── error_gold_180_with_checkpoint.csv  (180 rows from hasil_*/, enriched)
    ├── error_gold_180_all4.csv             (+ RoBERTa fresh + Fine-tuned columns)
    ├── model_comparison.csv / .png
    ├── best_model.txt
    ├── confidence_buckets_all_topics.csv
    ├── gold_f1_by_bucket_all_topics.csv
    └── FINAL_RESULTS.md                     (this file)
```
