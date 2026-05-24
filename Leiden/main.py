"""
ECR 2.0 Pipeline — CLI version.

Input/output sama seperti main_gui.py, hanya beda interface:
- Input  : argparse flags (bukan upload web UI)
- Output : file structure identik dengan main_gui.py
           (workdir/users_leaning.csv, workdir/leiden/, workdir/infomap/,
            workdir/diffusion_*.csv, workdir/comparison_summary.txt,
            workdir/visualizations/)

Contoh:
    python main.py --dataset data.csv
    python main.py --dataset data.csv --workdir hasil --topic-polarity positive_is_pro
    python main.py --dataset data.csv --no-diffusion --calibration --calibration-method temperature
"""

import argparse
import math
import os
import re
import sys
import time
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


def log(msg):
    print(msg, flush=True)


def parse_mentioned(raw):
    """Parse kolom mentioned: (name,USERNAME)//(id,USERID) pairs."""
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
    s = str(val).strip().strip('"').strip("'").strip("﻿")
    try:
        return str(int(float(s)))
    except (ValueError, OverflowError):
        return s if s else None


def build_interactions(df, get_username=None):
    """Build interactions DataFrame (src, dst, type, weight) dari raw data."""
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


def run_pipeline(config):
    """Jalankan pipeline penuh — mirror dari LeidenPipelineGUI.run_pipeline."""
    timings = {}
    total_start = time.time()
    outdir = config["workdir"]
    os.makedirs(outdir, exist_ok=True)

    # ── 1. Load data ──
    log("\n[STEP 1] Loading data...")
    start = time.time()

    data_path = config["dataset"]
    if not data_path or not os.path.exists(data_path):
        log(f"[ERROR] Dataset CSV not found: {data_path}")
        return None

    df = pd.read_csv(data_path)
    log(f"  Rows: {len(df):,}")
    log(f"  Columns: {list(df.columns)}")

    required = {"from_id", config["text_col"]}
    missing = required - set(df.columns)
    if missing:
        log(f"[ERROR] CSV missing required columns: {missing}")
        return None

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
    log(f"  Username map: {len(username_map):,} users")

    def get_username(uid):
        nid = normalize_id(uid)
        return username_map.get(nid, f"@{nid}")

    timings["Load data"] = time.time() - start
    log(f"[DONE] Data loaded. (Time: {timings['Load data']:.2f}s)")

    # ── 2. Sentiment inference ──
    log("\n[STEP 2] Fine-tuned sentiment inference...")
    start = time.time()

    proba_cols = tuple(config["proba_cols"].split(","))
    model_path = config["finetuned_model_path"]
    text_col = config["text_col"]
    batch_size = int(config["batch_size"])
    max_length = int(config["max_length"])

    log(f"  Model : {model_path}")
    log(f"  Texts : {len(df):,}  (col={text_col}, batch={batch_size}, max_len={max_length})")

    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    cache_path = os.path.join(outdir, f"{dataset_name}_sentiment_cache.csv")

    cache_valid = False
    if os.path.exists(cache_path):
        df_check = pd.read_csv(cache_path, nrows=5)
        if "indobert_neg" in df_check.columns or "negative" in df_check.columns:
            cache_valid = True

    if cache_valid:
        log(f"  Cache ditemukan: {cache_path}")
        df_sent = pd.read_csv(cache_path)
        log(f"  Loaded {len(df_sent):,} cached rows.")
    else:
        log(f"  No cache. Running inference (pertama kali, akan di-cache)...")

        is_lfs = False
        for lfs_check in ["tokenizer.json", "model.safetensors"]:
            check_file = os.path.join(model_path, lfs_check)
            if os.path.exists(check_file):
                with open(check_file, "rb") as _f:
                    header = _f.read(50)
                if b"git-lfs" in header:
                    is_lfs = True
                    break
        if is_lfs:
            model_path = "amaliap21/ecr2-sentiment-xlmroberta"
            log(f"  Local model is LFS pointer, downloading from HuggingFace: {model_path}")

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        except ImportError:
            log("[ERROR] pip install transformers torch")
            return None

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
            batch_num = (bi // batch_size) + 1
            if batch_num % 50 == 0 or done == len(text_list):
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log(f"  df_sent columns: {list(df_sent.columns)}")
    log(f"  df_sent shape: {df_sent.shape}")
    col_neg, col_neu, col_pos = proba_cols

    sent_col_map = {
        "negative": col_neg, "neg": col_neg, "indobert_neg": col_neg,
        "neutral": col_neu, "neu": col_neu, "indobert_neu": col_neu,
        "positive": col_pos, "pos": col_pos, "indobert_pos": col_pos,
    }
    label_col = next((c for c in df_sent.columns if "label" in c.lower()), None)

    for src_col in df_sent.columns:
        if src_col in sent_col_map:
            df[sent_col_map[src_col]] = df_sent[src_col].values
    if label_col:
        df["sentiment_label"] = df_sent[label_col].values

    if col_neg not in df.columns or col_pos not in df.columns:
        log(f"[ERROR] Sentiment columns not mapped. df_sent has: {list(df_sent.columns)}")
        return None

    timings["Sentiment inference"] = time.time() - start
    log(f"  Label distribution: {df_sent['indobert_label'].value_counts().to_dict()}")
    log(f"[DONE] Sentiment inference complete. (Time: {timings['Sentiment inference']:.2f}s)")

    # ── 3. Build interactions & graph ──
    log("\n[STEP 3] Building interactions & graph...")
    start = time.time()

    df_interactions = build_interactions(df, get_username=get_username)
    if len(df_interactions) == 0:
        log("[ERROR] No interactions found. Check reply_to_user_id / retweet_from_user_id / mentioned columns.")
        return None

    G, df_nodes = build_graph(df_interactions, directed=True)
    log(f"  Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}")

    timings["Build graph"] = time.time() - start
    log(f"[DONE] Graph built. (Time: {timings['Build graph']:.2f}s)")

    # ── 4. Estimate user leaning ──
    log("\n[STEP 4] Estimating user leaning...")
    start = time.time()

    topic_polarity = config["topic_polarity"]

    df_content = df[["from_id"] + list(proba_cols)].copy()
    df_content["user"] = df_content["from_id"].apply(get_username)
    df_content = df_content.drop(columns=["from_id"]).dropna(subset=["user"])

    df_users = estimate_user_leaning(
        df_content,
        user_col="user",
        proba_cols=proba_cols,
        topic_polarity=topic_polarity,
    )
    log(f"  Users with leaning: {len(df_users)}")

    timings["User leaning"] = time.time() - start
    log(f"[DONE] User leaning estimated. (Time: {timings['User leaning']:.2f}s)")

    # ── 5. Calibration (optional) ──
    if config["calibration_enabled"]:
        log("\n[STEP 5] Calibrating probabilities...")
        start = time.time()
        cal_method = config["calibration_method"]

        df_users = calibrate_probabilities(
            df_users,
            method=cal_method,
            proba_cols=("p_neg_mean", "p_neu_mean", "p_pos_mean"),
        )
        log(f"  Calibration method: {cal_method}")

        timings["Calibration"] = time.time() - start
        log(f"[DONE] Calibration complete. (Time: {timings['Calibration']:.2f}s)")
    else:
        log("\n[STEP 5] Calibration skipped.")

    # ── 6. Community detection (Leiden + Infomap) ──
    log("\n[STEP 6] Detecting communities (Leiden + Infomap)...")
    start = time.time()

    resolution = config["resolution"]
    if config["auto_resolution"]:
        resolution = max(0.5, math.log10(G.number_of_nodes() + 1))
        log(f"  Auto resolution: {resolution:.3f}")
    else:
        log(f"  Resolution: {resolution:.3f}")

    rs = int(config["random_state"])

    comms_louvain = detect_communities(G, method="louvain", random_state=rs)
    n_louvain = len(set(comms_louvain.values()))
    log(f"  Louvain : {n_louvain} communities (baseline, Aidira 2025)")

    comms_leiden = detect_communities(G, method="leiden", resolution=resolution, random_state=rs)
    n_leiden = len(set(comms_leiden.values()))
    log(f"  Leiden  : {n_leiden} communities")

    comms_infomap = detect_communities(G, method="infomap", random_state=rs)
    n_infomap = len(set(comms_infomap.values()))
    log(f"  Infomap : {n_infomap} communities")

    # Multi-Level Consensus Clustering (Pho dkk., 2021) over {Leiden, Infomap}.
    # Louvain is kept as a baseline comparison only, not in the MLCC ensemble.
    log("  MLCC (Pho 2021): running 7-step consensus on {Leiden, Infomap}...")
    comms_consensus = mlcc_consensus_partition(
        G,
        partitions=[comms_leiden, comms_infomap],
        n_levels=3,
        similarity_metric="cosine",
        resolution=resolution,
        random_state=rs,
        log_fn=log,
    )
    n_consensus = len(set(comms_consensus.values()))
    log(f"  Consensus (MLCC): {n_consensus} communities")

    timings["Community detection"] = time.time() - start
    log(f"[DONE] Community detection complete. (Time: {timings['Community detection']:.2f}s)")

    partitions_dict = {
        "Louvain": comms_louvain,
        "Leiden": comms_leiden,
        "Infomap": comms_infomap,
        "Consensus": comms_consensus,
    }

    # ── 6b. Quality metrics ──
    log("\n[STEP 6b] Validating community quality...")
    start_q = time.time()

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
            "modularity": mod,
            "coverage": coverage,
            "performance": performance,
            "mean_conductance": mean_cond,
        }
        log(f"\n  [{label}] Quality Metrics:")
        log(f"    Modularity     : {mod:.4f}  (>0.3 = meaningful structure)")
        log(f"    Coverage       : {coverage:.4f}  (fraction of intra-community edges)")
        log(f"    Performance    : {performance:.4f}  (correctly placed node pairs)")
        log(f"    Conductance    : {mean_cond:.4f}  (lower = better separated, <0.5 good)")

    try:
        from sklearn.metrics import normalized_mutual_info_score
        pair_labels = list(partitions_dict.keys())
        for i in range(len(pair_labels)):
            for j in range(i + 1, len(pair_labels)):
                la_, lb_ = pair_labels[i], pair_labels[j]
                ca, cb = partitions_dict[la_], partitions_dict[lb_]
                common_users = sorted(set(ca.keys()) & set(cb.keys()))
                if len(common_users) > 10:
                    labels_a = [ca[u] for u in common_users]
                    labels_b = [cb[u] for u in common_users]
                    nmi_score = normalized_mutual_info_score(labels_a, labels_b)
                    quality_metrics[f"nmi_{la_}_vs_{lb_}"] = nmi_score
                    log(f"\n  [NMI {la_}<->{lb_}]: {nmi_score:.4f}  (agreement between algorithms)")
                    if nmi_score > 0.5:
                        log(f"    -> Kedua partisi menemukan struktur komunitas yang serupa (NMI > 0.5)")
                    else:
                        log(f"    -> Perbedaan signifikan dalam partisi (NMI <= 0.5)")
    except ImportError:
        log("  WARN: sklearn not available, NMI skipped")

    log("\n  [VERDICT]")
    for label in partitions_dict.keys():
        qm = quality_metrics[label]
        reliable = qm["modularity"] > 0.3 and qm["coverage"] > 0.3 and qm["mean_conductance"] < 0.5
        status = "RELIABLE" if reliable else "PERLU DICEK"
        log(f"    {label}: {status} (mod={qm['modularity']:.3f}, cov={qm['coverage']:.3f}, cond={qm['mean_conductance']:.3f})")

    timings["Quality metrics"] = time.time() - start_q
    log(f"[DONE] Quality validation complete. (Time: {timings['Quality metrics']:.2f}s)")

    # ── 7. Compute ECR 2.0 ──
    log("\n[STEP 7] Computing ECR 2.0 (Leiden + Infomap + MLCC)...")
    start = time.time()

    results = {}
    for label, comms in partitions_dict.items():
        ecr2 = compute_ecr2(G, df_users, comms)
        results[label] = {
            "ecr2": ecr2, "communities": comms,
            "n": len(set(comms.values())),
            "quality": quality_metrics[label],
        }
        log(f"\n  [{label}]")
        log(f"    Intra agreement : {ecr2.intra:.4f}")
        log(f"    Inter agreement : {ecr2.inter:.4f}")
        log(f"    ECR 2.0 ratio   : {ecr2.ratio:.4f}")
        log(f"    Homophily r     : {ecr2.homophily_r:.4f}")

    timings["ECR 2.0"] = time.time() - start
    log(f"\n[DONE] ECR 2.0 computed. (Time: {timings['ECR 2.0']:.2f}s)")

    # ── 8. ECR threshold ──
    log("\n[STEP 8] Estimating ECR threshold (Leiden + Infomap)...")
    start = time.time()

    n_threshold_samples = int(config["n_threshold_samples"])
    for label in results:
        comms = results[label]["communities"]
        threshold = estimate_ecr_threshold(G, df_users, comms, n_samples=n_threshold_samples, seed=rs)
        classification = classify_echo_chamber(results[label]["ecr2"].ratio, threshold)
        results[label]["threshold"] = threshold
        results[label]["classification"] = classification
        log(f"  [{label}] Threshold: {threshold:.4f} -> {classification}")

    timings["ECR threshold"] = time.time() - start
    log(f"[DONE] Threshold estimated. (Time: {timings['ECR threshold']:.2f}s)")

    # ── 8b. Statistical validation ──
    log("\n[STEP 8b] ECR Statistical Validation...")
    start = time.time()

    n_boot = 100
    n_perm = 100
    rng = np.random.RandomState(rs)

    for label in results:
        comms = results[label]["communities"]
        ecr_observed = results[label]["ecr2"].ratio

        boot_ratios = []
        users_in_graph = [u for u in df_users["user"] if u in G]
        n_users = len(users_in_graph)
        for _ in range(n_boot):
            idx = rng.choice(n_users, size=n_users, replace=True)
            sampled_users = [users_in_graph[i] for i in idx]
            H = G.subgraph(sampled_users).copy()
            if H.number_of_edges() < 5:
                continue
            try:
                ecr_b = compute_ecr2(H, df_users, comms)
                boot_ratios.append(ecr_b.ratio)
            except Exception:
                pass

        if len(boot_ratios) >= 10:
            ci_lo = float(np.percentile(boot_ratios, 2.5))
            ci_hi = float(np.percentile(boot_ratios, 97.5))
        else:
            ci_lo, ci_hi = ecr_observed, ecr_observed

        perm_ratios = []
        lean_vals = df_users.set_index("user")["lean_scalar"].to_dict()
        user_list = list(lean_vals.keys())
        lean_array = np.array([lean_vals[u] for u in user_list])
        for _ in range(n_perm):
            shuffled = rng.permutation(lean_array)
            df_perm = df_users.copy()
            perm_map = dict(zip(user_list, shuffled))
            df_perm["lean_scalar"] = df_perm["user"].map(perm_map)
            try:
                ecr_p = compute_ecr2(G, df_perm, comms)
                perm_ratios.append(ecr_p.ratio)
            except Exception:
                pass

        if perm_ratios:
            p_value = float(np.mean([1 if pr <= ecr_observed else 0 for pr in perm_ratios]))
            p_value = max(1.0 / (len(perm_ratios) + 1), 1.0 - p_value)
        else:
            p_value = 1.0

        results[label]["bootstrap_ci"] = (ci_lo, ci_hi)
        results[label]["p_value"] = p_value

        threshold = results[label]["threshold"]
        log(f"\n  [{label}] Statistical Validation:")
        log(f"    ECR 2.0 ratio      : {ecr_observed:.4f}")
        log(f"    Bootstrap 95% CI   : [{ci_lo:.4f}, {ci_hi:.4f}]")
        log(f"    Permutation p-value: {p_value:.4f} {'(significant)' if p_value < 0.05 else '(not significant)'}")

        # Correct logic: ECR >= threshold means observed agreement structure
        # exceeds null-model expectation, i.e., echo chamber present.
        # (Cinelli et al., 2021; Amendola et al., 2024.)
        if ecr_observed >= threshold and p_value < 0.05 and ci_lo >= threshold:
            verdict = "ECHO CHAMBER DETECTED (strong evidence)"
        elif ecr_observed >= threshold and p_value < 0.05:
            verdict = "ECHO CHAMBER LIKELY (ECR >= threshold, p < 0.05)"
        elif ecr_observed >= threshold:
            verdict = "POSSIBLE ECHO CHAMBER (ECR >= threshold, but p >= 0.05)"
        else:
            verdict = "NO ECHO CHAMBER (ECR < threshold)"
        results[label]["verdict"] = verdict
        log(f"    -> {verdict}")

    timings["ECR validation"] = time.time() - start
    log(f"[DONE] ECR validation complete. (Time: {timings['ECR validation']:.2f}s)")

    # ── 9. Homophily ──
    log("\n[STEP 9] Computing homophily...")
    start = time.time()
    homo = compute_homophily(G, df_users)
    log(f"  Pearson r: {homo['pearson_r']:.4f}")
    timings["Homophily"] = time.time() - start
    log(f"[DONE] Homophily computed. (Time: {timings['Homophily']:.2f}s)")

    # ── 10. Diffusion bias ──
    if config["diffusion_enabled"]:
        log("\n[STEP 10] Simulating diffusion bias (IC)...")
        start = time.time()
        ic_df = simulate_diffusion_bias(
            G, df_users,
            p_activate=float(config["p_activate"]),
            n_runs=int(config["n_runs"]),
            seed=rs,
        )
        ic_summary = summarize_diffusion_bias(ic_df)
        log(f"  Seeds simulated: {len(ic_df)}")
        timings["Diffusion bias"] = time.time() - start
        log(f"[DONE] Diffusion simulation complete. (Time: {timings['Diffusion bias']:.2f}s)")
    else:
        ic_df = None
        ic_summary = None
        log("\n[STEP 10] Diffusion simulation skipped.")

    # ── 11. Community classification ──
    log("\n[STEP 11] Classifying communities...")
    start = time.time()
    for label in results:
        comms = results[label]["communities"]
        df_comm = classify_communities_detailed(G, df_users, comms)
        results[label]["df_comm"] = df_comm
        log(f"  [{label}] {len(df_comm)} communities classified")
    timings["Community classification"] = time.time() - start
    log(f"[DONE] Community classification complete. (Time: {timings['Community classification']:.2f}s)")

    # ── 12. Diffusion bias metrics ──
    if config["diffusion_enabled"] and ic_df is not None:
        log("\n[STEP 12] Computing diffusion bias metrics...")
        start = time.time()
        for label in results:
            comms = results[label]["communities"]
            try:
                dm = compute_diffusion_bias_metrics(ic_df, comms, df_users)
                results[label]["diff_metrics"] = dm
                gm = dm["global_metrics"]
                log(f"  [{label}] slope={gm['slope']:.4f}, R2={gm['r_squared']:.4f}, "
                    f"bias_ratio={gm['bias_ratio']:.4f}, Cohen_d={gm['cohens_d']:.4f}")
            except Exception as e:
                log(f"  [{label}] WARN: {e}")
                results[label]["diff_metrics"] = None
        timings["Diffusion metrics"] = time.time() - start
        log(f"[DONE] Diffusion metrics computed. (Time: {timings['Diffusion metrics']:.2f}s)")

    # ── 13. Save outputs ──
    log("\n[STEP 13] Saving outputs...")
    start = time.time()

    users_path = os.path.join(outdir, "users_leaning.csv")
    df_users.to_csv(users_path, index=False)
    log(f"  Saved: {users_path}")

    for label in results:
        r = results[label]
        prefix = label.lower()
        sub = os.path.join(outdir, prefix)
        os.makedirs(sub, exist_ok=True)

        comm_df = pd.DataFrame([{"user": u, "community": c} for u, c in r["communities"].items()])
        comm_df.to_csv(os.path.join(sub, "communities.csv"), index=False)

        r["df_comm"].to_csv(os.path.join(sub, "community_classification.csv"), index=False)

        ecr2 = r["ecr2"]
        with open(os.path.join(sub, "ecr_metrics.txt"), "w", encoding="utf-8") as f:
            f.write(f"ECR 2.0 Results — {label}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Intra-community agreement : {ecr2.intra:.6f}\n")
            f.write(f"Inter-community agreement : {ecr2.inter:.6f}\n")
            f.write(f"ECR 2.0 ratio             : {ecr2.ratio:.6f}\n")
            f.write(f"Homophily (Pearson r)      : {ecr2.homophily_r:.6f}\n")
            f.write(f"Threshold                  : {r['threshold']:.6f}\n")
            f.write(f"Classification             : {r['classification']}\n")
            f.write(f"Number of communities      : {r['n']}\n")
            f.write(f"Resolution                 : {resolution:.3f}\n")
            f.write(f"Topic polarity             : {topic_polarity}\n")
            qm = r.get("quality", {})
            if qm:
                f.write(f"\n--- Community Quality Metrics ---\n")
                f.write(f"Modularity                 : {qm['modularity']:.6f}\n")
                f.write(f"Coverage                   : {qm['coverage']:.6f}\n")
                f.write(f"Performance                : {qm['performance']:.6f}\n")
                f.write(f"Mean Conductance           : {qm['mean_conductance']:.6f}\n")
            bci = r.get("bootstrap_ci")
            pv = r.get("p_value")
            vd = r.get("verdict")
            if bci is not None:
                f.write(f"\n--- Statistical Validation ---\n")
                f.write(f"Bootstrap 95% CI           : [{bci[0]:.6f}, {bci[1]:.6f}]\n")
                f.write(f"Permutation p-value        : {pv:.6f}\n")
                f.write(f"Verdict                    : {vd}\n")

        dm = r.get("diff_metrics")
        if dm is not None:
            dm["community_metrics"].to_csv(os.path.join(sub, "diffusion_bias_metrics.csv"), index=False)

        log(f"  Saved: {sub}/")

    if ic_df is not None:
        ic_df.to_csv(os.path.join(outdir, "diffusion_ic.csv"), index=False)
        ic_summary.to_csv(os.path.join(outdir, "diffusion_summary.csv"), index=False)
        log(f"  Saved: diffusion_ic.csv, diffusion_summary.csv")

    summary_path = os.path.join(outdir, "comparison_summary.txt")
    labels_out = list(results.keys())
    col_w = 14
    header_w = 24 + col_w * len(labels_out)

    def _row(name, values):
        cells = "".join(f"{v:>{col_w}}" for v in values)
        return f"{name:<24}{cells}\n"

    def _row_num(name, values, fmt=".4f"):
        cells = "".join(f"{v:>{col_w}{fmt}}" for v in values)
        return f"{name:<24}{cells}\n"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*header_w}\n")
        f.write(f"  ECR 2.0 — Multi-Algorithm Comparison (Leiden, Infomap, MLCC)\n")
        f.write(f"{'='*header_w}\n\n")
        f.write(_row("Metric", labels_out))
        f.write("-" * header_w + "\n")
        f.write(_row("Communities", [results[l]["n"] for l in labels_out]))
        f.write(_row_num("Intra agreement", [results[l]["ecr2"].intra for l in labels_out]))
        f.write(_row_num("Inter agreement", [results[l]["ecr2"].inter for l in labels_out]))
        f.write(_row_num("ECR 2.0 ratio", [results[l]["ecr2"].ratio for l in labels_out]))
        f.write(_row_num("Threshold", [results[l]["threshold"] for l in labels_out]))
        f.write(_row("Classification", [results[l]["classification"] for l in labels_out]))

        f.write(f"\n{'='*header_w}\n")
        f.write(f"  Community Quality Validation\n")
        f.write(f"{'='*header_w}\n\n")
        f.write(_row("Metric", labels_out))
        f.write("-" * header_w + "\n")
        f.write(_row_num("Modularity", [results[l]["quality"].get("modularity", 0) for l in labels_out]))
        f.write(_row_num("Coverage", [results[l]["quality"].get("coverage", 0) for l in labels_out]))
        f.write(_row_num("Performance", [results[l]["quality"].get("performance", 0) for l in labels_out]))
        f.write(_row_num("Conductance (mean)",
                         [results[l]["quality"].get("mean_conductance", 0) for l in labels_out]))

        f.write("\n")
        for key, val in quality_metrics.items():
            if key.startswith("nmi_"):
                pair = key[4:].replace("_vs_", " <-> ")
                f.write(f"NMI ({pair}) : {val:.4f}\n")

        f.write(f"\n{'='*header_w}\n")
        f.write(f"  ECR Statistical Validation\n")
        f.write(f"{'='*header_w}\n\n")
        f.write(_row("Metric", labels_out))
        f.write("-" * header_w + "\n")
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
            f.write(f"Verdict {l:<10}: {results[l].get('verdict', 'N/A')}\n")

        f.write(f"\nHomophily (Pearson r) : {homo['pearson_r']:.4f}\n")
        f.write(f"Topic polarity        : {topic_polarity}\n")
        f.write(f"Resolution (Leiden)   : {resolution:.3f}\n")
        f.write(f"MLCC: 7-step (Pho dkk., 2021), ensemble={{Leiden, Infomap}}, "
                f"L=3 levels, cosine similarity\n")
    log(f"  Saved: {summary_path}")

    timings["Save outputs"] = time.time() - start
    log(f"[DONE] Outputs saved. (Time: {timings['Save outputs']:.2f}s)")

    # ── 14. Visualizations ──
    log("\n[STEP 14] Generating visualizations...")
    start = time.time()
    viz_dir = os.path.join(outdir, "visualizations")
    viz_paths = generate_all_visualizations(
        G, df_users, results, homo, ic_df, viz_dir,
        df=df, df_sent=df_sent, log_fn=log,
    )
    timings["Visualizations"] = time.time() - start
    log(f"[DONE] {len(viz_paths)} visualizations saved to {viz_dir}/ (Time: {timings['Visualizations']:.2f}s)")

    # ── Summary ──
    total_elapsed = time.time() - total_start
    log("\n" + "=" * 60)
    log("[PIPELINE COMPLETE]")
    log(f"Output directory: {os.path.abspath(outdir)}")
    log(f"\n[PIPELINE TIMINGS]")
    for comp, t in timings.items():
        log(f"  {comp}: {t:.2f}s")
    log(f"  Total: {total_elapsed:.2f}s")
    log("=" * 60)

    return results


def build_parser():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_finetuned = os.path.join(script_dir, "finetuned_sentiment_cardiff_xlmroberta")

    p = argparse.ArgumentParser(
        description="ECR 2.0 Pipeline (CLI) — sama input/output dengan main_gui.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset Input (= upload pada GUI)
    p.add_argument("--dataset", required=True,
                   help="Path ke CSV dataset. Wajib mengandung from_id, cleaned_text, "
                        "+ minimal satu dari reply_to_user_id / retweet_from_user_id / mentioned.")

    # General Settings
    p.add_argument("--workdir", default="pipeline_out",
                   help="Folder output untuk semua hasil pipeline.")
    p.add_argument("--random-state", type=int, default=42,
                   help="Seed untuk reproducibility.")

    # Sentiment Configuration
    p.add_argument("--finetuned-model-path", default=default_finetuned,
                   help="Path ke folder fine-tuned model.")
    p.add_argument("--text-col", default="cleaned_text",
                   help="Nama kolom CSV yang berisi teks.")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size inferensi sentiment.")
    p.add_argument("--max-length", type=int, default=128,
                   help="Panjang maksimum token per teks.")
    p.add_argument("--proba-cols", default="p_neg,p_neu,p_pos",
                   help="Nama 3 kolom output probabilitas (neg,neu,pos), dipisah koma.")

    # User Leaning
    p.add_argument("--topic-polarity", default="negative_is_pro",
                   choices=["negative_is_pro", "positive_is_pro"],
                   help="Pemetaan sentimen ke posisi user terhadap topik.")

    # Calibration
    cal_group = p.add_mutually_exclusive_group()
    cal_group.add_argument("--calibration", dest="calibration_enabled",
                           action="store_true", help="Aktifkan kalibrasi probabilitas.")
    cal_group.add_argument("--no-calibration", dest="calibration_enabled",
                           action="store_false", help="Matikan kalibrasi (default).")
    p.set_defaults(calibration_enabled=False)
    p.add_argument("--calibration-method", default="isotonic",
                   choices=["isotonic", "temperature"],
                   help="Metode kalibrasi.")

    # Community Detection
    res_group = p.add_mutually_exclusive_group()
    res_group.add_argument("--auto-resolution", dest="auto_resolution",
                           action="store_true",
                           help="Hitung resolution = max(0.5, log10(N+1)) (default).")
    res_group.add_argument("--manual-resolution", dest="auto_resolution",
                           action="store_false",
                           help="Pakai --resolution untuk nilai manual.")
    p.set_defaults(auto_resolution=True)
    p.add_argument("--resolution", type=float, default=1.0,
                   help="Resolution Leiden (dipakai jika --manual-resolution).")

    # Diffusion & Threshold
    diff_group = p.add_mutually_exclusive_group()
    diff_group.add_argument("--diffusion", dest="diffusion_enabled",
                            action="store_true", help="Aktifkan simulasi diffusion (default).")
    diff_group.add_argument("--no-diffusion", dest="diffusion_enabled",
                            action="store_false", help="Skip simulasi diffusion (lebih cepat).")
    p.set_defaults(diffusion_enabled=True)
    p.add_argument("--p-activate", type=float, default=0.05,
                   help="Probabilitas aktivasi per edge (Independent Cascade).")
    p.add_argument("--n-runs", type=int, default=20,
                   help="Jumlah simulasi IC per seed node.")
    p.add_argument("--n-threshold-samples", type=int, default=100,
                   help="Jumlah graf random untuk estimasi threshold ECR.")

    return p


def args_to_config(args):
    """Konversi argparse Namespace ke dict yang sama dengan config GUI."""
    return {
        "dataset": os.path.abspath(args.dataset),
        "workdir": args.workdir,
        "random_state": args.random_state,
        "finetuned_model_path": args.finetuned_model_path,
        "text_col": args.text_col,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "proba_cols": args.proba_cols,
        "topic_polarity": args.topic_polarity,
        "calibration_enabled": args.calibration_enabled,
        "calibration_method": args.calibration_method,
        "auto_resolution": args.auto_resolution,
        "resolution": args.resolution,
        "diffusion_enabled": args.diffusion_enabled,
        "p_activate": args.p_activate,
        "n_runs": args.n_runs,
        "n_threshold_samples": args.n_threshold_samples,
    }


def main():
    args = build_parser().parse_args()
    config = args_to_config(args)

    log("ECR 2.0 Pipeline (CLI)")
    log(f"  Dataset : {config['dataset']}")
    log(f"  Workdir : {os.path.abspath(config['workdir'])}")
    log(f"  Model   : {config['finetuned_model_path']}")
    log(f"  Topic polarity     : {config['topic_polarity']}")
    log(f"  Calibration        : {config['calibration_enabled']} ({config['calibration_method']})")
    log(f"  Auto resolution    : {config['auto_resolution']} (manual={config['resolution']})")
    log(f"  Diffusion          : {config['diffusion_enabled']} (p={config['p_activate']}, runs={config['n_runs']})")
    log(f"  Threshold samples  : {config['n_threshold_samples']}")
    log(f"  Random state       : {config['random_state']}")

    try:
        run_pipeline(config)
    except Exception:
        log(f"\n[ERROR] Pipeline failed:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
