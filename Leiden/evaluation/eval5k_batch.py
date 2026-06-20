"""
Helper to let Claude (this session) label the pool in batches, no API.

  python eval5k_batch.py --status            # progress per topic
  python eval5k_batch.py --next 150          # print next 150 unlabeled rows
  python eval5k_batch.py --apply _pending.tsv  # append labels to gold_labels.csv

--next prints lines:  <uid>\t<topic>\t<text>   (newlines stripped from text)
--apply reads lines:  <uid>\t<label>           (label: 1/2/3 or neg/neu/pos)

Rows are served in pool order, which is grouped by topic, so labeling stays in
one topic's stance frame at a time. Every applied batch is flushed to disk, so
the whole process is resumable.
"""

import argparse
import csv
import os
import sys

import pandas as pd

from eval5k_config import EVAL_DIR

# Windows console defaults to cp1252; force UTF-8 so emoji/special chars print.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

POOL = os.path.join(EVAL_DIR, "pool.csv")
GOLD = os.path.join(EVAL_DIR, "gold_labels.csv")
GOLD_FIELDS = ["uid", "topic", "manual_label"]

LABELMAP = {
    "1": "negative", "neg": "negative", "negative": "negative", "n": "negative",
    "2": "neutral",  "neu": "neutral",  "neutral": "neutral",  "e": "neutral",
    "3": "positive", "pos": "positive", "positive": "positive", "p": "positive",
}


def norm(text):
    """Normalized text key for exact-duplicate (retweet) matching."""
    return " ".join(str(text).split()).strip().lower()


def load_done():
    if not os.path.exists(GOLD):
        return set()
    return set(pd.read_csv(GOLD, usecols=["uid"])["uid"].astype(str))


def load_gold_df():
    if not os.path.exists(GOLD):
        return pd.DataFrame(columns=GOLD_FIELDS)
    return pd.read_csv(GOLD)


def unlabeled(pool):
    done = load_done()
    return pool[~pool["uid"].astype(str).isin(done)]


def append_rows(rows):
    """rows: list of dicts with uid/topic/manual_label. Append+flush."""
    new_file = not os.path.exists(GOLD)
    with open(GOLD, "a", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=GOLD_FIELDS)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow(r)
        out.flush()
        os.fsync(out.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--next", type=int, metavar="N")
    ap.add_argument("--apply", metavar="FILE")
    ap.add_argument("--apply-seq", metavar="FILE",
                    help="file of labels (one per line, 1/2/3) applied IN ORDER "
                         "to the next unlabeled rows (must match the last --next)")
    ap.add_argument("--topic", help="restrict --next/--status to one topic")
    ap.add_argument("--propagate", action="store_true",
                    help="copy each existing label to all unlabeled rows with "
                         "identical (normalized) text")
    args = ap.parse_args()

    pool = pd.read_csv(POOL)
    if args.topic:
        pool = pool[pool["topic"] == args.topic]

    if args.propagate:
        gold = load_gold_df()
        if gold.empty:
            print("no labels yet to propagate")
            return
        pool2 = pool.copy()
        pool2["_norm"] = pool2["text"].map(norm)
        # norm text -> label, from already-labeled rows (per topic, to be safe)
        lab = pool2.merge(gold[["uid", "manual_label"]], on="uid", how="inner")
        norm2label = {}
        for _, r in lab.iterrows():
            norm2label.setdefault((r["topic"], r["_norm"]), r["manual_label"])
        done = load_done()
        rows = []
        for _, r in pool2.iterrows():
            if str(r["uid"]) in done:
                continue
            key = (r["topic"], r["_norm"])
            if key in norm2label:
                rows.append({"uid": r["uid"], "topic": r["topic"],
                             "manual_label": norm2label[key]})
                done.add(str(r["uid"]))
        append_rows(rows)
        print(f"propagated {len(rows)} duplicate rows. total now {len(load_done()):,}")
        return

    if args.status:
        done = load_done()
        d = pool.assign(_d=pool["uid"].astype(str).isin(done))
        g = d.groupby("topic")["_d"].agg(["sum", "count"])
        total_done = int(d["_d"].sum())
        print(f"labeled {total_done:,}/{len(pool):,}")
        for t, r in g.iterrows():
            print(f"  {t:16s}: {int(r['sum']):,}/{int(r['count']):,}")
        return

    if args.next:
        todo = unlabeled(pool).copy()
        # serve one representative per UNIQUE normalized text (skip retweets
        # already represented in this batch). Apply the label, then --propagate
        # fans it out to the identical duplicates.
        todo["_norm"] = todo["text"].map(norm)
        todo = todo.drop_duplicates(subset="_norm", keep="first").head(args.next)
        for _, row in todo.iterrows():
            text = " ".join(str(row["text"]).split())
            sys.stdout.write(f"{row['uid']}\t{row['topic']}\t{text}\n")
        return

    if args.apply_seq:
        with open(args.apply_seq, encoding="utf-8") as f:
            tokens = f.read().split()
        labels = [LABELMAP.get(t.strip().lower()) for t in tokens]
        todo = unlabeled(pool).head(len(labels))
        if len(todo) != len(labels):
            sys.exit(f"have {len(labels)} labels but {len(todo)} unlabeled rows")
        bad = [i for i, l in enumerate(labels) if l is None]
        if bad:
            sys.exit(f"unrecognized label tokens at positions {bad[:10]}")
        new_file = not os.path.exists(GOLD)
        with open(GOLD, "a", newline="", encoding="utf-8") as out:
            w = csv.DictWriter(out, fieldnames=GOLD_FIELDS)
            if new_file:
                w.writeheader()
            for (_, row), label in zip(todo.iterrows(), labels):
                w.writerow({"uid": row["uid"], "topic": row["topic"],
                            "manual_label": label})
            out.flush()
            os.fsync(out.fileno())
        print(f"applied {len(labels)} (seq). total now "
              f"{len(load_done()):,}")
        return

    if args.apply:
        new_file = not os.path.exists(GOLD)
        done = load_done()
        applied = skipped = 0
        with open(args.apply, encoding="utf-8") as f, \
                open(GOLD, "a", newline="", encoding="utf-8") as out:
            w = csv.DictWriter(out, fieldnames=GOLD_FIELDS)
            if new_file:
                w.writeheader()
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                uid, raw = parts[0].strip(), parts[1].strip().lower()
                label = LABELMAP.get(raw)
                if uid in done or label is None:
                    skipped += 1
                    continue
                topic = uid.rsplit("_", 1)[0]
                w.writerow({"uid": uid, "topic": topic, "manual_label": label})
                done.add(uid)
                applied += 1
            out.flush()
            os.fsync(out.fileno())
        print(f"applied {applied}, skipped {skipped}. total now {len(done):,}")
        return

    ap.error("choose --status, --next N, or --apply FILE")


if __name__ == "__main__":
    main()
