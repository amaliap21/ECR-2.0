"""
Resumable model inference over the eval pool (outputs/eval5k/pool.csv).

Each model writes its own cache so runs are independent and restartable:
    pred_finetuned.csv    uid, finetuned_label, finetuned_confidence
    pred_indobert.csv     uid, indobert_label,  indobert_confidence
    pred_roberta_api.csv  uid, roberta_api_label, roberta_api_confidence
    pred_gpt.csv          uid, gpt_label

(roberta_api_cached_label needs no inference — it is carried in pool.csv.)

Every model skips uids already present in its cache, so you can stop/restart or
re-run after a crash and it continues. Predictions are flushed to disk every
batch.

Usage:
    python eval5k_infer.py --models finetuned indobert roberta_api      # local + free HTTP
    python eval5k_infer.py --models gpt                                 # OpenAI (costs $)
    python eval5k_infer.py --models all
"""

import argparse
import csv
import os
import sys
import time

import pandas as pd

from eval5k_config import EVAL_DIR, ROOT, norm_label

POOL_PATH = os.path.join(EVAL_DIR, "pool.csv")


# ----------------------------------------------------------------------------
# resumable-cache helpers
# ----------------------------------------------------------------------------
def cache_path(key):
    return os.path.join(EVAL_DIR, f"pred_{key}.csv")


def load_cache_uids(key):
    p = cache_path(key)
    if not os.path.exists(p):
        return set()
    return set(pd.read_csv(p, usecols=["uid"])["uid"].astype(str))


class CacheWriter:
    """Append rows to a per-model cache, flushing to disk each batch."""

    def __init__(self, key, fields):
        self.path = cache_path(key)
        self.fields = fields
        self.new = not os.path.exists(self.path)
        self.f = open(self.path, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        if self.new:
            self.w.writeheader()

    def write(self, rows):
        for r in rows:
            self.w.writerow(r)
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        self.f.close()


def pending(pool, key):
    have = load_cache_uids(key)
    todo = pool[~pool["uid"].astype(str).isin(have)].reset_index(drop=True)
    print(f"[{key}] {len(have):,} cached, {len(todo):,} to infer")
    return todo


# ----------------------------------------------------------------------------
# 1) Fine-tuned XLM-RoBERTa  (local model dir)
# ----------------------------------------------------------------------------
def infer_finetuned(pool, batch=32):
    todo = pending(pool, "finetuned")
    if todo.empty:
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    mdir = os.path.join(ROOT, "finetuned_sentiment")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  loading fine-tuned XLM-RoBERTa on {device} ...")
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(mdir).to(device).eval()
    id2label = {int(k): norm_label(v) for k, v in model.config.id2label.items()}

    cw = CacheWriter("finetuned", ["uid", "finetuned_label", "finetuned_confidence"])
    texts = todo["text"].fillna("").astype(str).tolist()
    uids = todo["uid"].tolist()
    try:
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                chunk = texts[i:i + batch]
                enc = tok(chunk, padding=True, truncation=True, max_length=256,
                          return_tensors="pt").to(device)
                probs = torch.softmax(model(**enc).logits, dim=-1)
                conf, idx = probs.max(dim=-1)
                rows = [{"uid": uids[i + j],
                         "finetuned_label": id2label[int(idx[j])],
                         "finetuned_confidence": float(conf[j])}
                        for j in range(len(chunk))]
                cw.write(rows)
                print(f"    finetuned {i + len(chunk):,}/{len(texts):,}", end="\r")
    finally:
        cw.close()
    print(f"\n  finetuned done.")


# ----------------------------------------------------------------------------
# 2) IndoBERT  (mdhugol/indonesia-bert-sentiment-classification)
# ----------------------------------------------------------------------------
def infer_indobert(pool, batch=32):
    todo = pending(pool, "indobert")
    if todo.empty:
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    name = "mdhugol/indonesia-bert-sentiment-classification"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  loading IndoBERT ({name}) on {device} ...")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
    id2label = {int(k): norm_label(v) for k, v in model.config.id2label.items()}

    cw = CacheWriter("indobert", ["uid", "indobert_label", "indobert_confidence"])
    texts = [str(t)[:512] for t in todo["text"].fillna("").tolist()]
    uids = todo["uid"].tolist()
    try:
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                chunk = texts[i:i + batch]
                enc = tok(chunk, padding=True, truncation=True, max_length=128,
                          return_tensors="pt").to(device)
                probs = torch.softmax(model(**enc).logits, dim=-1)
                conf, idx = probs.max(dim=-1)
                rows = [{"uid": uids[i + j],
                         "indobert_label": id2label.get(int(idx[j]), "neutral"),
                         "indobert_confidence": float(conf[j])}
                        for j in range(len(chunk))]
                cw.write(rows)
                print(f"    indobert {i + len(chunk):,}/{len(texts):,}", end="\r")
    finally:
        cw.close()
    print(f"\n  indobert done.")


# ----------------------------------------------------------------------------
# 3) RoBERTa API ("IndoBERT API No Limit") — self-hosted HTTP endpoint
# ----------------------------------------------------------------------------
def infer_roberta_api(pool, batch=20):
    todo = pending(pool, "roberta_api")
    if todo.empty:
        return
    import requests
    URL = "http://103.67.43.247:9443/sentiment/infer-sentiment-roberta"

    def call(texts):
        payload = {"list": [{"text": t} for t in texts]}
        for attempt in range(3):
            try:
                r = requests.post(URL, json=payload, timeout=60)
                r.raise_for_status()
                return r.json()["list"]
            except Exception as e:
                print(f"\n    retry {attempt + 1}: {e}")
                time.sleep(2 * (attempt + 1))
        raise RuntimeError("RoBERTa API failed (endpoint may be down)")

    cw = CacheWriter("roberta_api", ["uid", "roberta_api_label", "roberta_api_confidence"])
    texts = todo["text"].fillna("").astype(str).tolist()
    uids = todo["uid"].tolist()
    try:
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            res = call(chunk)
            rows = [{"uid": uids[i + j],
                     "roberta_api_label": norm_label(res[j].get("label")),
                     "roberta_api_confidence": res[j].get("probability")}
                    for j in range(len(chunk))]
            cw.write(rows)
            print(f"    roberta_api {i + len(chunk):,}/{len(texts):,}", end="\r")
    finally:
        cw.close()
    print(f"\n  roberta_api done.")


# ----------------------------------------------------------------------------
# 4) GPT  (OpenAI, OPENAI_MODEL from .env) — batched JSON to cut cost
# ----------------------------------------------------------------------------
def _load_env():
    from dotenv import dotenv_values
    env = dotenv_values(os.path.join(ROOT, ".env"))
    return {k: v for k, v in env.items() if v is not None}


def infer_gpt(pool, batch=20):
    todo = pending(pool, "gpt")
    if todo.empty:
        return
    import json
    import openai

    env = _load_env()
    openai.api_key = env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    model = env.get("OPENAI_MODEL", "gpt-5.4-mini")
    if not openai.api_key:
        sys.exit("OPENAI_API_KEY not found in .env")
    print(f"  GPT model = {model}")

    SYS = ("Anda pengklasifikasi sentimen teks media sosial Bahasa Indonesia. "
           "Untuk setiap teks, tentukan sentimen: negative, neutral, atau positive. "
           "Balas HANYA dengan array JSON berisi label, urut sesuai input, "
           'contoh: ["negative","neutral","positive"]. Tanpa penjelasan.')

    def classify(texts):
        numbered = "\n".join(f"{i+1}. {t[:600]}" for i, t in enumerate(texts))
        for attempt in range(4):
            try:
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "system", "content": SYS},
                              {"role": "user", "content": numbered}],
                )
                txt = resp["choices"][0]["message"]["content"].strip()
                s, e = txt.find("["), txt.rfind("]")
                labels = json.loads(txt[s:e + 1])
                labels = [norm_label(x) for x in labels]
                if len(labels) == len(texts):
                    return labels
                # length mismatch -> pad/trim defensively
                labels = (labels + ["neutral"] * len(texts))[:len(texts)]
                return labels
            except Exception as ex:
                print(f"\n    gpt retry {attempt + 1}: {ex}")
                time.sleep(3 * (attempt + 1))
        # last resort: neutral, so the run can continue (rare)
        return ["neutral"] * len(texts)

    cw = CacheWriter("gpt", ["uid", "gpt_label"])
    texts = todo["text"].fillna("").astype(str).tolist()
    uids = todo["uid"].tolist()
    try:
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            labels = classify(chunk)
            rows = [{"uid": uids[i + j], "gpt_label": labels[j]} for j in range(len(chunk))]
            cw.write(rows)
            print(f"    gpt {i + len(chunk):,}/{len(texts):,}", end="\r")
    finally:
        cw.close()
    print(f"\n  gpt done.")


RUNNERS = {
    "finetuned": infer_finetuned,
    "indobert": infer_indobert,
    "roberta_api": infer_roberta_api,
    "gpt": infer_gpt,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    choices=list(RUNNERS) + ["all"])
    ap.add_argument("--limit", type=int, default=0,
                    help="only infer first N pool rows (for smoke testing)")
    args = ap.parse_args()

    if not os.path.exists(POOL_PATH):
        sys.exit(f"pool not found: {POOL_PATH} (run eval5k_build_pool.py first)")
    pool = pd.read_csv(POOL_PATH)
    if args.limit:
        pool = pool.head(args.limit)

    models = list(RUNNERS) if "all" in args.models else args.models
    for m in models:
        print(f"\n=== {m} ===")
        RUNNERS[m](pool)


if __name__ == "__main__":
    main()
