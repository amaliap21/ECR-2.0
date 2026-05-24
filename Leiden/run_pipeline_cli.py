"""
CLI runner for the ECR 2.0 pipeline.
Runs all 14 steps (sentiment → graph → communities → ECR → visualizations)
for a given dataset CSV and outputs everything to pipeline_out/.
"""

import os
import sys
import re
import time
import math
import json
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecr2_pipeline import (
    build_graph,
    estimate_user_leaning,
    calibrate_probabilities,
    detect_communities,
    mlcc_consensus_partition,
    compute_homophily,
    simulate_diffusion_bias,
    summarize_diffusion_bias,
    compute_ecr2,
    estimate_ecr_threshold,
    classify_echo_chamber,
    classify_communities_detailed,
    compute_diffusion_bias_metrics,
)
from visualizations import generate_all_visualizations

# ── Config ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Prefer the latest Cardiff XLM-RoBERTa fine-tuned model if present.
_MODEL_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "finetuned_sentiment_cardiff_xlmroberta"),
    os.path.join(SCRIPT_DIR, "finetuned_sentiment"),
]
MODEL_PATH = next((p for p in _MODEL_CANDIDATES if os.path.isdir(p)), _MODEL_CANDIDATES[-1])
OUTDIR = os.path.join(SCRIPT_DIR, "pipeline_out")

DATASETS = {
    "boikot": os.path.join(SCRIPT_DIR, "total_data_cleaned_mcd.csv"),
    "vaksin": os.path.join(SCRIPT_DIR, "total_data_cleaned_vaksin.csv"),
    "indonesia_gelap": os.path.join(SCRIPT_DIR, "total_data_cleaned_indonesia_gelap.csv"),
}

TEXT_COL = "cleaned_text"
PROBA_COLS = ("p_neg", "p_neu", "p_pos")
TOPIC_POLARITY = "negative_is_pro"
BATCH_SIZE = 64
MAX_LENGTH = 128
SEED = 42


def log(msg):
    print(msg, flush=True)


def parse_mentioned(raw):
    if pd.isna(raw):
        return []
    items = []
    for pair in str(raw).split("//"):
        m_id = re.search(r"\(id,([^)]+)\)", pair)
        m_name = re.search(r"\(name,([^)]+)\)", pair)
        if m_id:
            items.append({
                "user_id": str(m_id.group(1)).strip(),
                "username": m_name.group(1).strip() if m_name else None,
            })
    return items


def normalize_id(val):
    if pd.isna(val):
        return None
    s = str(val).strip().strip('"').strip("'").strip("\ufeff")
    try:
        return str(int(float(s)))
    except (ValueError, OverflowError):
        return s if s else None


def build_interactions(df, get_username=None):
    for col in ["from_id", "reply_to_user_id", "retweet_from_user_id"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_id)

    content_users = set(df["from_id"].dropna().unique())
    parts = []

    if "reply_to_user_id" in df.columns:
        r = df.loc[df["reply_to_user_id"].notna(), ["from_id", "reply_to_user_id"]].copy()
        r.columns = ["src", "dst"]
        r = r.dropna()
        r["type"] = "reply"
        r["weight"] = 1
        parts.append(r)
        log(f"  Reply edges: {len(r):,}")

    if "retweet_from_user_id" in df.columns:
        r = df.loc[df["retweet_from_user_id"].notna(), ["from_id", "retweet_from_user_id"]].copy()
        r.columns = ["src", "dst"]
        r = r.dropna()
        r["type"] = "retweet"
        r["weight"] = 1
        parts.append(r)
        log(f"  Retweet edges: {len(r):,}")

    if "mentioned" in df.columns:
        md = []
        for _, row in df[df["mentioned"].notna()].iterrows():
            for item in parse_mentioned(row["mentioned"]):
                if item["user_id"] and item["user_id"] != row["from_id"]:
                    md.append({
                        "src": row["from_id"],
                        "dst": item["user_id"],
                        "type": "mention",
                        "weight": 1,
                    })
        if md:
            parts.append(pd.DataFrame(md))
            log(f"  Mention edges: {len(md):,}")

    if not parts:
        return pd.DataFrame(columns=["src", "dst", "type", "weight"])

    interactions = pd.concat(parts, ignore_index=True)
    interactions = interactions[interactions["src"] != interactions["dst"]]
    before = len(interactions)
    interactions = interactions[
        interactions["src"].isin(content_users) & interactions["dst"].isin(content_users)
    ]
    log(f"  Filtered: {before:,} -> {len(interactions):,} (users with content only)")

    if get_username is not None:
        interactions["src"] = interactions["src"].apply(get_username)
        interactions["dst"] = interactions["dst"].apply(get_username)

    return interactions


def run_sentiment(df, model_path, text_col, batch_size, max_length, cache_path):
    if os.path.exists(cache_path):
        log(f"  Cache found: {cache_path}")
        return pd.read_csv(cache_path)

    log(f"  No cache. Running inference...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    log(f"  id2label: {id2label}")

    text_list = [str(t)[:512] for t in df[text_col].fillna("")]
    all_probs, all_labels = [], []

    for bi in range(0, len(text_list), batch_size):
        batch = text_list[bi:bi + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)

        for j in range(len(batch)):
            label = id2label.get(int(preds[j]), "neutral")
            all_labels.append(label)
            prob_map = {id2label.get(idx, "neutral"): float(p) for idx, p in enumerate(probs[j])}
            all_probs.append(prob_map)

        done = min(bi + batch_size, len(text_list))
        if (bi // batch_size) % 50 == 0 or done == len(text_list):
            log(f"  Progress: {done:,}/{len(text_list):,} ({100*done/len(text_list):.1f}%)")

    df_sent = pd.DataFrame({
        "indobert_label": all_labels,
        "indobert_neg": [p.get("negative", 0.0) for p in all_probs],
        "indobert_neu": [p.get("neutral", 0.0) for p in all_probs],
        "indobert_pos": [p.get("positive", 0.0) for p in all_probs],
    })
    df_sent.to_csv(cache_path, index=False)
    log(f"  Cached to: {cache_path}")

    del model, tokenizer
    import torch as _torch
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()

    return df_sent


def run_pipeline(data_path, topic_name, outdir):
    timings = {}
    total_start = time.time()
    os.makedirs(outdir, exist_ok=True)

    # ── 1. Load data ──
    log(f"\n[STEP 1] Loading data: {topic_name}...")
    start = time.time()
    df = pd.read_csv(data_path)
    log(f"  Rows: {len(df):,}")

    username_map = {}
    if "from_username" in df.columns:
        _u = df[["from_id", "from_username"]].dropna(subset=["from_username"]).drop_duplicates("from_id")
        username_map.update(dict(zip(_u["from_id"].apply(normalize_id), _u["from_username"])))
    if "mentioned" in df.columns:
        for _, row in df[df["mentioned"].notna()].iterrows():
            for item in parse_mentioned(row["mentioned"]):
                uid = normalize_id(item["user_id"])
                if uid and uid not in username_map and item.get("username"):
                    username_map[uid] = item["username"]

    def get_username(uid):
        nid = normalize_id(uid)
        return username_map.get(nid, f"@{nid}")

    timings["Load data"] = time.time() - start
    log(f"[DONE] Data loaded. ({timings['Load data']:.2f}s)")

    # ── 2. Sentiment inference ──
    log("\n[STEP 2] Sentiment inference (fine-tuned model)...")
    start = time.time()
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    cache_path = os.path.join(outdir, f"{dataset_name}_sentiment_cache_xlmroberta.csv")
    log(f"  Model: {MODEL_PATH}")

    df_sent = run_sentiment(df, MODEL_PATH, TEXT_COL, BATCH_SIZE, MAX_LENGTH, cache_path)

    col_neg, col_neu, col_pos = PROBA_COLS
    df[col_neg] = df_sent["indobert_neg"].values
    df[col_neu] = df_sent["indobert_neu"].values
    df[col_pos] = df_sent["indobert_pos"].values
    df["sentiment_label"] = df_sent["indobert_label"].values

    timings["Sentiment"] = time.time() - start
    log(f"  Labels: {df_sent['indobert_label'].value_counts().to_dict()}")
    log(f"[DONE] Sentiment. ({timings['Sentiment']:.2f}s)")

    # ── 3. Build graph ──
    log("\n[STEP 3] Building graph...")
    start = time.time()
    df_interactions = build_interactions(df, get_username=get_username)
    if len(df_interactions) == 0:
        log("[ERROR] No interactions found."); return
    G, df_nodes = build_graph(df_interactions, directed=True)
    log(f"  Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}")
    timings["Build graph"] = time.time() - start
    log(f"[DONE] ({timings['Build graph']:.2f}s)")

    # ── 4. User leaning ──
    log("\n[STEP 4] Estimating user leaning...")
    start = time.time()
    df_content = df[["from_id"] + list(PROBA_COLS)].copy()
    df_content["user"] = df_content["from_id"].apply(get_username)
    df_content = df_content.drop(columns=["from_id"]).dropna(subset=["user"])
    df_users = estimate_user_leaning(df_content, user_col="user",
                                     proba_cols=PROBA_COLS, topic_polarity=TOPIC_POLARITY)
    log(f"  Users: {len(df_users)}")
    timings["User leaning"] = time.time() - start
    log(f"[DONE] ({timings['User leaning']:.2f}s)")

    # ── 5. Calibration (skip) ──
    log("\n[STEP 5] Calibration skipped.")

    # ── 6. Community detection ──
    log("\n[STEP 6] Detecting communities (Louvain + Leiden + Infomap + MLCC consensus)...")
    start = time.time()
    resolution = max(0.5, math.log10(G.number_of_nodes() + 1))
    log(f"  Auto resolution: {resolution:.3f}")

    comms_louvain = detect_communities(G, method="louvain", random_state=SEED)
    n_louvain = len(set(comms_louvain.values()))
    log(f"  Louvain: {n_louvain} communities (baseline, Aidira 2025)")

    comms_leiden = detect_communities(G, method="leiden", resolution=resolution, random_state=SEED)
    n_leiden = len(set(comms_leiden.values()))
    log(f"  Leiden:  {n_leiden} communities")

    comms_infomap = detect_communities(G, method="infomap", random_state=SEED)
    n_infomap = len(set(comms_infomap.values()))
    log(f"  Infomap: {n_infomap} communities")

    # Multi-Level Consensus Clustering (Pho dkk., 2021) — 7-step framework
    # over the {Leiden, Infomap} ensemble. Louvain is kept as a baseline
    # comparison only, not part of the consensus ensemble.
    log("  MLCC (Pho 2021): running 7-step consensus on {Leiden, Infomap}...")
    comms_consensus = mlcc_consensus_partition(
        G,
        partitions=[comms_leiden, comms_infomap],
        n_levels=3,
        similarity_metric="cosine",
        resolution=resolution,
        random_state=SEED,
        log_fn=log,
    )
    n_consensus = len(set(comms_consensus.values()))
    log(f"  Consensus (MLCC): {n_consensus} communities")
    timings["Communities"] = time.time() - start
    log(f"[DONE] ({timings['Communities']:.2f}s)")

    partitions_dict = {
        "Louvain": comms_louvain,
        "Leiden": comms_leiden,
        "Infomap": comms_infomap,
        "Consensus": comms_consensus,
    }

    # ── 6b. Quality metrics ──
    log("\n[STEP 6b] Quality validation...")
    start = time.time()
    from networkx.algorithms.community.quality import modularity as nx_modularity, partition_quality
    from collections import defaultdict

    G_undirected = G.to_undirected() if G.is_directed() else G
    quality_metrics = {}
    for label, comms in partitions_dict.items():
        comm_sets = defaultdict(set)
        for user, cid in comms.items():
            if user in G_undirected:
                comm_sets[cid].add(user)
        partition = list(comm_sets.values())

        mod = nx_modularity(G_undirected, partition, weight="weight")
        coverage, performance = partition_quality(G_undirected, partition)
        conductances = []
        for cs in partition:
            if len(cs) < 2:
                continue
            internal = boundary = 0
            for u in cs:
                for v in G_undirected.neighbors(u):
                    if v in cs:
                        internal += 1
                    else:
                        boundary += 1
            denom = 2 * internal + boundary
            if denom > 0:
                conductances.append(boundary / denom)
        mean_cond = float(np.mean(conductances)) if conductances else 0.0

        quality_metrics[label] = {
            "modularity": mod, "coverage": coverage,
            "performance": performance, "mean_conductance": mean_cond,
        }
        log(f"  [{label}] mod={mod:.4f} cov={coverage:.4f} cond={mean_cond:.4f}")

    try:
        from sklearn.metrics import normalized_mutual_info_score
        pair_labels = list(partitions_dict.keys())
        for i in range(len(pair_labels)):
            for j in range(i + 1, len(pair_labels)):
                la_, lb_ = pair_labels[i], pair_labels[j]
                ca, cb = partitions_dict[la_], partitions_dict[lb_]
                common = sorted(set(ca) & set(cb))
                if len(common) > 10:
                    nmi = normalized_mutual_info_score(
                        [ca[u] for u in common],
                        [cb[u] for u in common])
                    quality_metrics[f"nmi_{la_}_vs_{lb_}"] = nmi
                    log(f"  NMI {la_} vs {lb_}: {nmi:.4f}")
    except ImportError:
        pass
    timings["Quality"] = time.time() - start
    log(f"[DONE] ({timings['Quality']:.2f}s)")

    # ── 7. ECR 2.0 ──
    log("\n[STEP 7] Computing ECR 2.0...")
    start = time.time()
    results = {}
    for label, comms in partitions_dict.items():
        ecr2 = compute_ecr2(G, df_users, comms)
        results[label] = {
            "ecr2": ecr2, "communities": comms,
            "n": len(set(comms.values())), "quality": quality_metrics[label],
        }
        log(f"  [{label}] intra={ecr2.intra:.4f} inter={ecr2.inter:.4f} ratio={ecr2.ratio:.4f}")
    timings["ECR"] = time.time() - start
    log(f"[DONE] ({timings['ECR']:.2f}s)")

    # ── 8. Threshold + classification ──
    log("\n[STEP 8] ECR threshold...")
    start = time.time()
    for label in results:
        comms = results[label]["communities"]
        threshold = estimate_ecr_threshold(G, df_users, comms, n_samples=100, seed=SEED)
        classification = classify_echo_chamber(results[label]["ecr2"].ratio, threshold)
        results[label]["threshold"] = threshold
        results[label]["classification"] = classification
        log(f"  [{label}] threshold={threshold:.4f} -> {classification}")
    timings["Threshold"] = time.time() - start
    log(f"[DONE] ({timings['Threshold']:.2f}s)")

    # ── 8b. Statistical validation ──
    log("\n[STEP 8b] Statistical validation (bootstrap + permutation)...")
    start = time.time()
    rng = np.random.RandomState(SEED)
    for label in results:
        comms = results[label]["communities"]
        ecr_obs = results[label]["ecr2"].ratio

        users_in_graph = [u for u in df_users["user"] if u in G]
        n_users = len(users_in_graph)
        boot_ratios = []
        for _ in range(100):
            idx = rng.choice(n_users, size=n_users, replace=True)
            H = G.subgraph([users_in_graph[i] for i in idx]).copy()
            if H.number_of_edges() < 5:
                continue
            try:
                boot_ratios.append(compute_ecr2(H, df_users, comms).ratio)
            except Exception:
                pass
        ci_lo = float(np.percentile(boot_ratios, 2.5)) if len(boot_ratios) >= 10 else ecr_obs
        ci_hi = float(np.percentile(boot_ratios, 97.5)) if len(boot_ratios) >= 10 else ecr_obs

        lean_vals = df_users.set_index("user")["lean_scalar"].to_dict()
        user_list = list(lean_vals.keys())
        lean_array = np.array([lean_vals[u] for u in user_list])
        perm_ratios = []
        for _ in range(100):
            shuffled = rng.permutation(lean_array)
            df_perm = df_users.copy()
            df_perm["lean_scalar"] = df_perm["user"].map(dict(zip(user_list, shuffled)))
            try:
                perm_ratios.append(compute_ecr2(G, df_perm, comms).ratio)
            except Exception:
                pass
        if perm_ratios:
            p_value = max(1.0 / (len(perm_ratios) + 1),
                          1.0 - float(np.mean([1 if pr <= ecr_obs else 0 for pr in perm_ratios])))
        else:
            p_value = 1.0

        results[label]["bootstrap_ci"] = (ci_lo, ci_hi)
        results[label]["p_value"] = p_value

        threshold = results[label]["threshold"]
        # Correct logic: ECR >= threshold means observed agreement structure
        # exceeds null-model expectation, i.e., echo chamber present.
        # (Cinelli et al., 2021; Amendola et al., 2024.)
        if ecr_obs >= threshold and p_value < 0.05 and ci_lo >= threshold:
            verdict = "ECHO CHAMBER DETECTED (strong evidence)"
        elif ecr_obs >= threshold and p_value < 0.05:
            verdict = "ECHO CHAMBER LIKELY (ECR >= threshold, p < 0.05)"
        elif ecr_obs >= threshold:
            verdict = "POSSIBLE ECHO CHAMBER (ECR >= threshold, but p >= 0.05)"
        else:
            verdict = "NO ECHO CHAMBER (ECR < threshold)"
        results[label]["verdict"] = verdict
        log(f"  [{label}] CI=[{ci_lo:.4f},{ci_hi:.4f}] p={p_value:.4f} -> {verdict}")
    timings["Validation"] = time.time() - start
    log(f"[DONE] ({timings['Validation']:.2f}s)")

    # ── 9. Homophily ──
    log("\n[STEP 9] Homophily...")
    start = time.time()
    homo = compute_homophily(G, df_users)
    log(f"  Pearson r: {homo['pearson_r']:.4f}")
    timings["Homophily"] = time.time() - start

    # ── 10. Diffusion bias ──
    log("\n[STEP 10] Diffusion simulation...")
    start = time.time()
    ic_df = simulate_diffusion_bias(G, df_users, p_activate=0.05, n_runs=20, seed=SEED)
    ic_summary = summarize_diffusion_bias(ic_df)
    log(f"  Seeds: {len(ic_df)}")
    timings["Diffusion"] = time.time() - start
    log(f"[DONE] ({timings['Diffusion']:.2f}s)")

    # ── 11. Community classification ──
    log("\n[STEP 11] Classifying communities...")
    start = time.time()
    for label in results:
        df_comm = classify_communities_detailed(G, df_users, results[label]["communities"])
        results[label]["df_comm"] = df_comm
        log(f"  [{label}] {len(df_comm)} communities")
    timings["Classification"] = time.time() - start

    # ── 12. Diffusion metrics ──
    log("\n[STEP 12] Diffusion bias metrics...")
    start = time.time()
    for label in results:
        try:
            dm = compute_diffusion_bias_metrics(ic_df, results[label]["communities"], df_users)
            results[label]["diff_metrics"] = dm
            gm = dm["global_metrics"]
            log(f"  [{label}] slope={gm['slope']:.4f} R2={gm['r_squared']:.4f} "
                f"bias={gm['bias_ratio']:.4f} d={gm['cohens_d']:.4f}")
        except Exception as e:
            log(f"  [{label}] WARN: {e}")
            results[label]["diff_metrics"] = None
    timings["Diff metrics"] = time.time() - start

    # ── 13. Save outputs ──
    log("\n[STEP 13] Saving outputs...")
    start = time.time()
    df_users.to_csv(os.path.join(outdir, "users_leaning.csv"), index=False)

    for label in results:
        r = results[label]
        sub = os.path.join(outdir, label.lower())
        os.makedirs(sub, exist_ok=True)

        pd.DataFrame([{"user": u, "community": c} for u, c in r["communities"].items()]).to_csv(
            os.path.join(sub, "communities.csv"), index=False)
        r["df_comm"].to_csv(os.path.join(sub, "community_classification.csv"), index=False)

        ecr2 = r["ecr2"]
        with open(os.path.join(sub, "ecr_metrics.txt"), "w", encoding="utf-8") as f:
            f.write(f"ECR 2.0 Results — {label}\n{'='*50}\n")
            f.write(f"Intra-community agreement : {ecr2.intra:.6f}\n")
            f.write(f"Inter-community agreement : {ecr2.inter:.6f}\n")
            f.write(f"ECR 2.0 ratio             : {ecr2.ratio:.6f}\n")
            f.write(f"Homophily (Pearson r)      : {ecr2.homophily_r:.6f}\n")
            f.write(f"Threshold                  : {r['threshold']:.6f}\n")
            f.write(f"Classification             : {r['classification']}\n")
            f.write(f"Communities                : {r['n']}\n")
            f.write(f"Resolution                 : {resolution:.3f}\n")
            f.write(f"Topic polarity             : {TOPIC_POLARITY}\n")
            qm = r.get("quality", {})
            if qm:
                f.write(f"\n--- Quality ---\n")
                f.write(f"Modularity   : {qm['modularity']:.6f}\n")
                f.write(f"Coverage     : {qm['coverage']:.6f}\n")
                f.write(f"Performance  : {qm['performance']:.6f}\n")
                f.write(f"Conductance  : {qm['mean_conductance']:.6f}\n")
            bci = r.get("bootstrap_ci")
            if bci:
                f.write(f"\n--- Validation ---\n")
                f.write(f"Bootstrap CI : [{bci[0]:.6f}, {bci[1]:.6f}]\n")
                f.write(f"p-value      : {r['p_value']:.6f}\n")
                f.write(f"Verdict      : {r['verdict']}\n")

        dm = r.get("diff_metrics")
        if dm is not None:
            dm["community_metrics"].to_csv(os.path.join(sub, "diffusion_bias_metrics.csv"), index=False)

    ic_df.to_csv(os.path.join(outdir, "diffusion_ic.csv"), index=False)
    ic_summary.to_csv(os.path.join(outdir, "diffusion_summary.csv"), index=False)

    summary_path = os.path.join(outdir, "comparison_summary.txt")
    labels_out = list(results.keys())
    col_w = 14

    def _row(name, values):
        cells = "".join(f"{v:>{col_w}}" for v in values)
        return f"{name:<24}{cells}\n"

    def _row_num(name, values, fmt=".4f"):
        cells = "".join(f"{v:>{col_w}{fmt}}" for v in values)
        return f"{name:<24}{cells}\n"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*70}\n  ECR 2.0 — Multi-Algorithm Comparison (Leiden, Infomap, MLCC)\n{'='*70}\n\n")
        f.write(_row("Metric", labels_out))
        f.write("-" * (24 + col_w * len(labels_out)) + "\n")
        f.write(_row("Communities", [results[l]["n"] for l in labels_out]))
        f.write(_row_num("Intra agreement", [results[l]["ecr2"].intra for l in labels_out]))
        f.write(_row_num("Inter agreement", [results[l]["ecr2"].inter for l in labels_out]))
        f.write(_row_num("ECR 2.0 ratio", [results[l]["ecr2"].ratio for l in labels_out]))
        f.write(_row_num("Threshold", [results[l]["threshold"] for l in labels_out]))
        f.write(_row("Classification", [results[l]["classification"] for l in labels_out]))

        f.write(f"\n{'='*70}\n  Community Quality Validation\n{'='*70}\n\n")
        f.write(_row("Metric", labels_out))
        f.write("-" * (24 + col_w * len(labels_out)) + "\n")
        f.write(_row_num("Modularity", [results[l]["quality"]["modularity"] for l in labels_out]))
        f.write(_row_num("Coverage", [results[l]["quality"]["coverage"] for l in labels_out]))
        f.write(_row_num("Performance", [results[l]["quality"]["performance"] for l in labels_out]))
        f.write(_row_num("Conductance (mean)", [results[l]["quality"]["mean_conductance"] for l in labels_out]))

        f.write("\n")
        for key, val in quality_metrics.items():
            if key.startswith("nmi_"):
                pair = key[4:].replace("_vs_", " <-> ")
                f.write(f"NMI ({pair}) : {val:.4f}\n")

        f.write(f"\n{'='*70}\n  ECR Statistical Validation\n{'='*70}\n\n")
        f.write(_row("Metric", labels_out))
        f.write("-" * (24 + col_w * len(labels_out)) + "\n")
        f.write(_row_num("Bootstrap CI low",
                         [results[l].get("bootstrap_ci", (float('nan'), float('nan')))[0]
                          for l in labels_out]))
        f.write(_row_num("Bootstrap CI high",
                         [results[l].get("bootstrap_ci", (float('nan'), float('nan')))[1]
                          for l in labels_out]))
        f.write(_row_num("Permutation p-value",
                         [results[l].get("p_value", float('nan')) for l in labels_out]))

        f.write("\n")
        for l in labels_out:
            f.write(f"Verdict {l:<9}: {results[l].get('verdict', 'N/A')}\n")

        f.write(f"\nHomophily (Pearson r) : {homo['pearson_r']:.4f}\n")
        f.write(f"Topic polarity        : {TOPIC_POLARITY}\n")
        f.write(f"Resolution (Leiden)   : {resolution:.3f}\n")
        f.write(f"MLCC: 7-step (Pho dkk., 2021), ensemble={{Leiden, Infomap}}, "
                f"L=3 levels, cosine similarity\n")

    timings["Save"] = time.time() - start
    log(f"[DONE] ({timings['Save']:.2f}s)")

    # ── 14. Visualizations ──
    log("\n[STEP 14] Generating visualizations...")
    start = time.time()
    viz_dir = os.path.join(outdir, "visualizations")
    viz_paths = generate_all_visualizations(
        G, df_users, results, homo, ic_df, viz_dir,
        df=df, df_sent=df_sent, log_fn=log,
    )
    timings["Visualizations"] = time.time() - start
    log(f"[DONE] {len(viz_paths)} visualizations. ({timings['Visualizations']:.2f}s)")

    total = time.time() - total_start
    log(f"\n{'='*60}\n[PIPELINE COMPLETE] {topic_name}")
    log(f"Output: {os.path.abspath(outdir)}")
    for k, v in timings.items():
        log(f"  {k}: {v:.2f}s")
    log(f"  Total: {total:.2f}s\n{'='*60}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="boikot",
                        choices=list(DATASETS.keys()),
                        help="Topic to run (default: boikot)")
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: pipeline_out/<topic>_xlmroberta)")
    args = parser.parse_args()

    data_path = DATASETS[args.topic]
    outdir = args.outdir or os.path.join(OUTDIR, f"{args.topic}_xlmroberta")

    log(f"Running ECR pipeline: topic={args.topic}")
    log(f"  Data: {data_path}")
    log(f"  Model: {MODEL_PATH}")
    log(f"  Output: {outdir}")

    run_pipeline(data_path, args.topic, outdir)
