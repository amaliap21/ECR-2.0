"""
Shared config for the 5k-per-topic time-based evaluation of the 3 NEW topics
(korupsi, ijazah, mbg). The 3 original topics (vaksin, indonesia_gelap, boikot)
are already evaluated in outputs/summary/error_gold_180_all4.csv and are merged
back in at the compute_f1 step.

Pipeline of scripts (run in order):
  eval5k_build_pool.py  -> outputs/eval5k/pool.csv         (15,000 time-based rows)
  eval5k_label.py       -> outputs/eval5k/gold_labels.csv  (your manual gold, resumable)
  eval5k_infer.py       -> outputs/eval5k/pred_<model>.csv (5 models, resumable)
  eval5k_assemble.py    -> outputs/eval5k/eval_new3.csv  +  outputs/summary/eval_all6.csv
  05_compute_f1.py      -> outputs/summary/model_comparison*.csv  (all 6 topics)
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                     # .../Leiden
EVAL_DIR = os.path.join(HERE, "outputs", "eval5k")
SUMMARY_DIR = os.path.join(HERE, "outputs", "summary")

# How many time-based rows to sample per NEW topic.
SAMPLES_PER_TOPIC = 5000

# All 6 topics -> source cleaned CSV. The evaluation is 5,000 time-based rows
# per topic (30,000 total); the old 180 error-sample eval is no longer used.
# "boikot" (Boikot Produk Israel) lives in the mcd export.
TOPIC_SOURCES = {
    "vaksin":          os.path.join(ROOT, "total_data_cleaned_vaksin.csv"),
    "indonesia_gelap": os.path.join(ROOT, "total_data_cleaned_indonesia_gelap.csv"),
    "boikot":          os.path.join(ROOT, "total_data_cleaned_mcd.csv"),
    "korupsi":         os.path.join(ROOT, "total_data_cleaned_korupsi.csv"),
    "ijazah":          os.path.join(ROOT, "total_data_cleaned_ijazah.csv"),
    "mbg":             os.path.join(ROOT, "total_data_cleaned_mbg.csv"),
}

# Timestamp format in date_created, e.g. "18/05/2026 15.09.18"
DATE_FMT = "%d/%m/%Y %H.%M.%S"

# Canonical 3-class label space.
LABELS = ["negative", "neutral", "positive"]

# The 5 models -> the prediction column each one writes. Same column names the
# existing 180-sample evaluation (error_gold_180_all4.csv) uses, so the two
# datasets merge cleanly for compute_f1.
MODEL_COLUMNS = {
    "finetuned":    "finetuned_label",          # Fine-tuned XLM-RoBERTa (local model dir)
    "roberta_api":  "indobert_api_label",       # "IndoBERT API (No Limit)" HTTP endpoint
    "roberta_cached": "roberta_api_cached_label",  # cached final_sentiment (carried from source)
    "gpt":          "gpt_label",                # GPT (OpenAI, OPENAI_MODEL)
    "indobert":     "indobert_label",           # mdhugol/indonesia-bert-sentiment-classification
}

# Per-row stable id: "<topic>_<00000>".
def make_uid(topic, i):
    return f"{topic}_{i:05d}"


# Normalize any sentiment spelling (incl. Indonesian + HF LABEL_x) -> canonical.
_LABEL_NORM = {
    "positive": "positive", "pos": "positive", "pro": "positive",
    "positif": "positive", "label_0": "positive",
    "negative": "negative", "neg": "negative", "contra": "negative",
    "negatif": "negative", "label_2": "negative",
    "neutral": "neutral", "neu": "neutral", "netral": "neutral",
    "label_1": "neutral",
}


def norm_label(s):
    if not isinstance(s, str):
        return None
    return _LABEL_NORM.get(s.strip().lower(), s.strip().lower())


def ensure_dirs():
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)
