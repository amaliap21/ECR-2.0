"""
Run Aidira's (2025) Louvain + GDM consensus pipeline on the 3 new datasets
(ijazah, korupsi, mbg) by INVOKING HER CLI ENTRY POINTS via subprocess:

    community              (src/gdm/community.py)
    absa_community_merge   (src/gdm/absa_community_merge.py)
    consensus              (src/gdm/consensus.py)

No line of Aidira's source code is modified — this script only prepares
input CSVs in her expected schema and calls her CLIs. The ABSA component
(Gemini) is substituted by reusing IndSight's `final_sentiment` and
`sentiment_confidence_level` columns from the raw CSV, converted to a
3-class probability vector using

    p_main  = sentiment_confidence_level
    p_other = (1 - p_main) / 2

which matches the values found in the existing
`pipeline_out_louvain/{boikot,vaksin,indonesia_gelap}/absa_community.csv`
that the user produced earlier with the same substitution strategy.

Outputs under `pipeline_out_louvain/<topic>/`:

    community_input.csv     Aidira's schema (id, text, name,
                            in_reply_to_screen_name, in_reply_to_status_id)
    aspect.csv              id, aspect_category, sentiment_prob
                            (input for absa_community_merge)
    community_out.csv       written by `community` CLI
    community_graph.gml     written by `community` CLI
    absa_community.csv      written by `absa_community_merge` CLI
    consensus.txt           written by `consensus` CLI
    consensus_details.txt   written by `consensus` CLI
    community_sizes.png     bar chart of community sizes (top 20)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
GDM_SRC = os.path.join(REPO_ROOT, "Louvain", "src")
GDM_COMMUNITY_MOD = "gdm.community"
GDM_ABSA_MERGE_MOD = "gdm.absa_community_merge"
GDM_CONSENSUS_MOD = "gdm.consensus"
GDM_ENV = os.environ.copy()
GDM_ENV["PYTHONPATH"] = (
    GDM_SRC
    if not GDM_ENV.get("PYTHONPATH")
    else f"{GDM_SRC}{os.pathsep}{GDM_ENV['PYTHONPATH']}"
)


# ---------------- helpers ----------------


def label_to_3class(label, confidence):
    if pd.isna(confidence):
        confidence = 1.0 / 3.0
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 1.0 / 3.0
    confidence = max(0.0, min(1.0, confidence))
    p_main = confidence
    p_other = (1.0 - p_main) / 2.0
    label = str(label).strip().lower() if label is not None else ""
    if label in ("negative", "neg", "kontra"):
        return [p_main, p_other, p_other]
    if label in ("positive", "pos", "pro"):
        return [p_other, p_other, p_main]
    return [p_other, p_main, p_other]


def normalize_id(val):
    if pd.isna(val):
        return ""
    s = str(val).strip()
    s = s.strip('"').strip("'").lstrip("﻿")
    if not s:
        return ""
    try:
        return str(int(float(s)))
    except (ValueError, OverflowError):
        return s


def run_cmd(cmd, desc, log=print, env=None):
    log(f"  [CMD] {desc}: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        log(f"  [ERR] {desc} failed (rc={res.returncode})")
        log(res.stderr.strip())
        raise RuntimeError(f"{desc} failed")
    if res.stdout.strip():
        for line in res.stdout.strip().splitlines()[-6:]:
            log(f"        | {line}")


# ---------------- per-topic runner ----------------


def run_for_topic(topic, raw_csv, outdir, min_members=5, log=print):
    os.makedirs(outdir, exist_ok=True)
    log(f"\n=== {topic.upper()} ===")
    log(f"  raw_csv : {raw_csv}")
    log(f"  outdir  : {outdir}")

    df = pd.read_csv(raw_csv, low_memory=False)
    log(f"  rows    : {len(df):,}")

    # 1. community_input.csv (Aidira's schema) -------------------------------
    # NOTE: we intentionally OMIT the `text` column. Aidira's community.py
    # treats `text` as an optional column (`if "text" in available_columns`),
    # and her downstream `absa_community_merge.py` reads with Python's
    # default csv reader which raises `_csv.Error: field larger than field
    # limit (131072)` whenever a single merged-text row exceeds 128 KB —
    # exactly what happens after Aidira's `merge_graph_by_user` concatenates
    # retweet bodies inside a large community. Dropping `text` keeps the
    # graph rules (which only use reply_status_id / reply_screen_name /
    # name) intact and avoids touching her source code.
    cinput = pd.DataFrame({
        "id": df["original_id"].apply(normalize_id),
        "name": df["from_id"].apply(normalize_id),
        "in_reply_to_screen_name":
            df["reply_to_user_id"].apply(normalize_id) if "reply_to_user_id" in df.columns else "",
        "in_reply_to_status_id":
            df["reply_to_original_id"].apply(normalize_id) if "reply_to_original_id" in df.columns else "",
    })
    cinput = cinput[cinput["id"] != ""].drop_duplicates(subset="id")
    cinput_path = os.path.join(outdir, "community_input.csv")
    cinput.to_csv(cinput_path, index=False)
    log(f"  community_input.csv : {len(cinput):,} rows")

    # 2. Call Aidira's `community` CLI ---------------------------------------
    community_out = os.path.join(outdir, "community_out.csv")
    graph_out = os.path.join(outdir, "community_graph.gml")
    run_cmd([
        sys.executable,
        "-m", GDM_COMMUNITY_MOD,
        cinput_path,
        "--algo", "louvain",
        "--out-csv", community_out,
        "-o", graph_out,
        "--format", "gml",
    ], "community (Louvain)", log=log, env=GDM_ENV)

    # 3. aspect.csv : id, aspect_category, sentiment_prob --------------------
    # Single aspect named "sentiment"; sentiment_prob is the 3-vector
    # converted from (final_sentiment, sentiment_confidence_level).
    aspect_rows = []
    fs_col = "final_sentiment" if "final_sentiment" in df.columns else None
    cl_col = "sentiment_confidence_level" if "sentiment_confidence_level" in df.columns else None
    for _, row in df.iterrows():
        rid = normalize_id(row.get("original_id"))
        if not rid:
            continue
        label = row[fs_col] if fs_col else "neutral"
        conf = row[cl_col] if cl_col else 1.0 / 3.0
        vec = label_to_3class(label, conf)
        aspect_rows.append({
            "id": rid,
            "aspect_category": "sentiment",
            "sentiment_prob": str(vec),
        })
    aspect_path = os.path.join(outdir, "aspect.csv")
    pd.DataFrame(aspect_rows).drop_duplicates(subset=["id", "aspect_category"]).to_csv(
        aspect_path, index=False)
    log(f"  aspect.csv : {len(aspect_rows):,} rows")

    # 4. Call Aidira's `absa_community_merge` CLI ----------------------------
    absa_out = os.path.join(outdir, "absa_community.csv")
    run_cmd([
        sys.executable,
        "-m", GDM_ABSA_MERGE_MOD,
        "--aspect", aspect_path,
        "--meta", community_out,
        "--output", absa_out,
    ], "absa_community_merge", log=log, env=GDM_ENV)

    # 5. Call Aidira's `consensus` CLI ---------------------------------------
    consensus_out = os.path.join(outdir, "consensus.txt")
    details_out = os.path.join(outdir, "consensus_details.txt")
    run_cmd([
        sys.executable,
        "-m", GDM_CONSENSUS_MOD,
        absa_out,
        "-o", consensus_out,
        "--details", details_out,
        "--min-members", str(min_members),
    ], "consensus (GDM)", log=log, env=GDM_ENV)

    # 6. Parse outputs for summary --------------------------------------------
    info = {"topic": topic}
    try:
        cons_text = Path(consensus_out).read_text(encoding="utf-8")
        import re
        m_w = re.search(r"WITHIN-COMMUNITY CONSENSUS ===\s*Mean = ([0-9.]+)", cons_text)
        m_b = re.search(r"BETWEEN-COMMUNITY CONSENSUS ===\s*Mean = ([0-9.]+)", cons_text)
        m_e = re.search(r"ECR \(Mean\) = ([0-9.]+)", cons_text)
        info["within"] = float(m_w.group(1)) if m_w else float("nan")
        info["between"] = float(m_b.group(1)) if m_b else float("nan")
        info["ecr"] = float(m_e.group(1)) if m_e else float("nan")
        log(f"  within={info['within']:.4f} between={info['between']:.4f} "
            f"ECR={info['ecr']:.4f}")
    except Exception as e:
        log(f"  [WARN] could not parse consensus.txt: {e}")

    # Count communities used (>= min_members) from absa_community.csv
    try:
        absa = pd.read_csv(absa_out)
        sizes = absa["community"].value_counts()
        info["communities_used"] = int((sizes >= min_members).sum())
        info["communities_total"] = int(sizes.size)
        log(f"  communities used (>={min_members}): {info['communities_used']:,} "
            f"of {info['communities_total']:,}")

        # community_sizes.png
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            top = sizes[sizes >= min_members].sort_values(ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(range(len(top)), top.values, color="steelblue")
            ax.set_xticks(range(len(top)))
            ax.set_xticklabels([str(c) for c in top.index], rotation=60,
                               ha="right", fontsize=7)
            ax.set_ylabel("Members")
            ax.set_title(f"{topic} — Louvain (Aidira) community sizes (top 20)")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "community_sizes.png"), dpi=120)
            plt.close(fig)
            log("  saved community_sizes.png")
        except ImportError:
            pass
    except Exception as e:
        log(f"  [WARN] could not summarise absa_community.csv: {e}")

    return info


# ---------------- driver --------------------------------------------------


DEFAULT_TOPICS = {
    "ijazah":  "total_data_cleaned_ijazah.csv",
    "korupsi": "total_data_cleaned_korupsi.csv",
    "mbg":     "total_data_cleaned_mbg.csv",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", nargs="*", default=list(DEFAULT_TOPICS),
                        help=f"Topics to run. Defaults: {list(DEFAULT_TOPICS)}")
    parser.add_argument("--outroot", default=os.path.join(THIS_DIR, "pipeline_out_louvain"),
                        help="Root output directory")
    parser.add_argument("--min-members", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.outroot, exist_ok=True)
    summaries = []
    for topic in args.topics:
        raw_name = DEFAULT_TOPICS.get(topic)
        if raw_name is None:
            print(f"[ERR] unknown topic '{topic}'")
            continue
        raw_csv = os.path.join(THIS_DIR, raw_name)
        if not os.path.isfile(raw_csv):
            print(f"[ERR] missing raw CSV: {raw_csv}")
            continue
        outdir = os.path.join(args.outroot, topic)
        try:
            info = run_for_topic(topic, raw_csv, outdir,
                                 min_members=args.min_members)
            summaries.append(info)
        except Exception as e:
            print(f"[ERR] topic '{topic}' failed: {e}")

    if summaries:
        print("\n=== ALL TOPICS DONE ===")
        rows = []
        for s in summaries:
            print(f"  {s['topic']:<10} ECR={s.get('ecr', float('nan')):.4f} "
                  f"within={s.get('within', float('nan')):.4f} "
                  f"between={s.get('between', float('nan')):.4f} "
                  f"comms_used={s.get('communities_used', '?')}")
            rows.append({
                "topic": s["topic"],
                "within_cc": round(s.get("within", float("nan")), 4),
                "between_cc": round(s.get("between", float("nan")), 4),
                "ecr": round(s.get("ecr", float("nan")), 4),
                "num_communities": s.get("communities_used", 0),
            })

        cmp_path = os.path.join(args.outroot, "comparison_summary.csv")
        df_new = pd.DataFrame(rows)
        if os.path.isfile(cmp_path):
            df_old = pd.read_csv(cmp_path)
            df_old = df_old[~df_old["topic"].isin(df_new["topic"])]
            df_out = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_out = df_new
        df_out.to_csv(cmp_path, index=False)
        print(f"  comparison_summary.csv updated: {cmp_path}")


if __name__ == "__main__":
    main()
