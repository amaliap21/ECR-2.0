"""
Interactive gold-label collector for the 5k-per-topic pool.

- Reads outputs/eval5k/pool.csv (15,000 rows over korupsi/ijazah/mbg).
- For each unlabeled row it shows the text and asks you for the sentiment.
- Every answer is appended to outputs/eval5k/gold_labels.csv IMMEDIATELY
  (write + flush + fsync), so the result is saved in real time and the session
  is fully resumable: re-run any time and it continues where you left off.

Controls per row:
    1 / n / neg      -> negative
    2 / e / neu      -> neutral
    3 / p / pos      -> positive
    s                -> skip (ask again next run)
    u                -> undo the last saved label
    q                -> quit (progress already saved)

By default labeling is BLIND (model predictions hidden) so your gold is not
biased by the models. Pass --show-preds to reveal them, or --topic <name> to
label one topic at a time.

Usage:
    python eval5k_label.py                 # all 3 topics, blind
    python eval5k_label.py --topic korupsi
    python eval5k_label.py --show-preds
"""

import argparse
import csv
import os
import sys

import pandas as pd

from eval5k_config import EVAL_DIR, ensure_dirs, LABELS

POOL_PATH = os.path.join(EVAL_DIR, "pool.csv")
GOLD_PATH = os.path.join(EVAL_DIR, "gold_labels.csv")
GOLD_FIELDS = ["uid", "topic", "manual_label"]

# Input keystroke -> canonical label.
KEYMAP = {
    "1": "negative", "n": "negative", "neg": "negative", "negative": "negative",
    "2": "neutral",  "e": "neutral",  "neu": "neutral",  "neutral": "neutral",
    "3": "positive", "p": "positive", "pos": "positive", "positive": "positive",
}


def load_done():
    """uid -> label for everything already saved (and the ordered uid list)."""
    if not os.path.exists(GOLD_PATH):
        return {}, []
    done, order = {}, []
    with open(GOLD_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("uid"):
                done[row["uid"]] = row.get("manual_label")
                order.append(row["uid"])
    return done, order


def append_gold(uid, topic, label):
    """Append one labeled row and force it to disk right away."""
    new_file = not os.path.exists(GOLD_PATH)
    with open(GOLD_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=GOLD_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({"uid": uid, "topic": topic, "manual_label": label})
        f.flush()
        os.fsync(f.fileno())


def rewrite_gold(rows):
    """Rewrite the whole gold file (used by undo). rows: list of dicts."""
    tmp = GOLD_PATH + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=GOLD_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in GOLD_FIELDS})
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, GOLD_PATH)


def undo_last():
    if not os.path.exists(GOLD_PATH):
        print("  (nothing to undo)")
        return
    with open(GOLD_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("  (nothing to undo)")
        return
    removed = rows.pop()
    rewrite_gold(rows)
    print(f"  undone: {removed['uid']} ({removed['manual_label']})")


PRED_COLS = ["finetuned_label", "indobert_label", "roberta_api_label",
             "roberta_api_cached_label", "gpt_label"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", help="label only this topic (korupsi/ijazah/mbg)")
    ap.add_argument("--show-preds", action="store_true",
                    help="reveal model predictions while labeling (biases gold)")
    args = ap.parse_args()

    ensure_dirs()
    if not os.path.exists(POOL_PATH):
        sys.exit(f"pool not found: {POOL_PATH}  (run eval5k_build_pool.py first)")

    pool = pd.read_csv(POOL_PATH)
    if args.topic:
        pool = pool[pool["topic"] == args.topic].reset_index(drop=True)
        if pool.empty:
            sys.exit(f"no rows for topic '{args.topic}'")

    # Optionally attach predictions for display.
    # (cache file key, label column inside that cache)
    pred_sources = [("finetuned", "finetuned_label"),
                    ("indobert", "indobert_label"),
                    ("roberta_api", "roberta_api_label"),
                    ("gpt", "gpt_label")]
    pred_lookup = {}
    if args.show_preds:
        # cached RoBERTa label lives in the pool itself
        pred_lookup["roberta_api_cached_label"] = dict(
            zip(pool["uid"], pool.get("roberta_api_cached_label", pd.Series(dtype=object))))
        for key, col in pred_sources:
            cache = os.path.join(EVAL_DIR, f"pred_{key}.csv")
            if os.path.exists(cache):
                cdf = pd.read_csv(cache)
                pred_lookup[col] = dict(zip(cdf["uid"], cdf[col]))

    done, _ = load_done()
    todo = pool[~pool["uid"].isin(done.keys())].reset_index(drop=True)

    total = len(pool)
    done_here = len(pool) - len(todo)
    print("=" * 64)
    print(f"  Gold labeling — {total:,} rows in scope, {done_here:,} done, "
          f"{len(todo):,} remaining")
    by_topic = pool.assign(_d=pool["uid"].isin(done.keys())).groupby("topic")["_d"].agg(["sum", "count"])
    for t, r in by_topic.iterrows():
        print(f"    {t:10s}: {int(r['sum']):,}/{int(r['count']):,}")
    print("  keys: 1/neg  2/neu  3/pos  |  s=skip  u=undo  q=quit")
    print("=" * 64)

    labeled_this_session = 0
    for _, row in todo.iterrows():
        uid, topic, text = row["uid"], row["topic"], row["text"]
        pos = done_here + labeled_this_session + 1
        print(f"\n[{pos:,}/{total:,}]  {topic}  ({uid})")
        print("-" * 64)
        print(text)
        if args.show_preds and pred_lookup:
            hints = "  ".join(f"{c.replace('_label',''):>13}={pred_lookup.get(c, {}).get(uid, '?')}"
                              for c in PRED_COLS)
            print("-" * 64)
            print("  preds:", hints)
        print("-" * 64)

        while True:
            try:
                ans = input("  sentiment [1/2/3, s/u/q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  quitting — progress saved.")
                return
            if ans == "q":
                print("  quitting — progress saved.")
                return
            if ans == "s":
                print("  skipped.")
                break
            if ans == "u":
                undo_last()
                done, _ = load_done()  # refresh
                break
            if ans in KEYMAP:
                label = KEYMAP[ans]
                append_gold(uid, topic, label)
                labeled_this_session += 1
                print(f"  saved: {label}")
                break
            print(f"  ? unrecognized '{ans}'. use 1/2/3, s, u, or q.")

    print(f"\nDone. Labeled {labeled_this_session} this session. "
          f"Gold file: {GOLD_PATH}")


if __name__ == "__main__":
    main()
