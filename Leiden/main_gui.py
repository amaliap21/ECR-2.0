import os
import sys
import traceback
import threading
import queue
import time
import math

import numpy as np
import pandas as pd
from nicegui import ui

# Add parent directory so ecr2_pipeline is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualizations import generate_all_visualizations

from ecr2_pipeline import (
    build_graph,
    estimate_user_leaning,
    calibrate_probabilities,
    detect_communities,
    compute_homophily,
    simulate_diffusion_bias,
    summarize_diffusion_bias,
    compute_ecr2,
    estimate_ecr_threshold,
    classify_echo_chamber,
    classify_communities_detailed,
    compute_diffusion_bias_metrics,
)


class LeidenPipelineGUI:
    def __init__(self):
        self.log_queue = queue.Queue()
        self.is_running = False
        self.uploaded_files = {}
        self.timings = {}
        self.viz_paths = []

    def log(self, message):
        self.log_queue.put(message)

    def handle_file_upload(self, e, file_type):
        try:
            uploads_dir = "uploads"
            os.makedirs(uploads_dir, exist_ok=True)
            upload_path = os.path.join(uploads_dir, e.name)
            with open(upload_path, 'wb') as f:
                f.write(e.content.read())
            self.uploaded_files[file_type] = upload_path
            self.log(f"[INFO] Uploaded {file_type}: {e.name}")
        except Exception as ex:
            self.log(f"[ERROR] Failed to upload {file_type}: {str(ex)}")

    @staticmethod
    def _parse_mentioned(raw):
        """Parse kolom mentioned: (name,USERNAME)//(id,USERID) pairs."""
        import re
        items = []
        if pd.isna(raw):
            return items
        for pair in str(raw).split('//'):
            m_id = re.search(r'\(id,([^)]+)\)', pair)
            m_name = re.search(r'\(name,([^)]+)\)', pair)
            if m_id:
                items.append({
                    'user_id': str(m_id.group(1)).strip(),
                    'username': m_name.group(1).strip() if m_name else None,
                })
        return items

    @staticmethod
    def _normalize_id(val):
        if pd.isna(val):
            return None
        s = str(val).strip().strip('"').strip("'").strip('\ufeff')
        try:
            return str(int(float(s)))
        except (ValueError, OverflowError):
            return s if s else None

    def _build_interactions(self, df, get_username=None):
        """Build interactions DataFrame (src, dst, type, weight) from raw data.
        If get_username is provided, node IDs are converted to usernames.
        """
        for col in ['from_id', 'reply_to_user_id', 'retweet_from_user_id']:
            if col in df.columns:
                df[col] = df[col].apply(self._normalize_id)

        content_users = set(df['from_id'].dropna().unique())
        interactions_list = []

        # Reply edges
        if 'reply_to_user_id' in df.columns:
            r = df.loc[df['reply_to_user_id'].notna(), ['from_id', 'reply_to_user_id']].copy()
            r.columns = ['src', 'dst']
            r = r.dropna()
            r['type'] = 'reply'
            r['weight'] = 1
            interactions_list.append(r)
            self.log(f"  Reply edges: {len(r):,}")

        # Retweet edges
        if 'retweet_from_user_id' in df.columns:
            r = df.loc[df['retweet_from_user_id'].notna(), ['from_id', 'retweet_from_user_id']].copy()
            r.columns = ['src', 'dst']
            r = r.dropna()
            r['type'] = 'retweet'
            r['weight'] = 1
            interactions_list.append(r)
            self.log(f"  Retweet edges: {len(r):,}")

        # Mention edges
        if 'mentioned' in df.columns:
            md = []
            for _, row in df[df['mentioned'].notna()].iterrows():
                for item in self._parse_mentioned(row['mentioned']):
                    if item['user_id'] and item['user_id'] != row['from_id']:
                        md.append({
                            'src': row['from_id'],
                            'dst': item['user_id'],
                            'type': 'mention',
                            'weight': 1,
                        })
            if md:
                interactions_list.append(pd.DataFrame(md))
                self.log(f"  Mention edges: {len(md):,}")

        if not interactions_list:
            return pd.DataFrame(columns=['src', 'dst', 'type', 'weight'])

        interactions = pd.concat(interactions_list, ignore_index=True)
        interactions = interactions[interactions['src'] != interactions['dst']]
        before = len(interactions)
        interactions = interactions[
            interactions['src'].isin(content_users) & interactions['dst'].isin(content_users)
        ]
        self.log(f"  Filtered: {before:,} -> {len(interactions):,} (users with content only)")

        # Convert IDs to usernames
        if get_username is not None:
            interactions['src'] = interactions['src'].apply(get_username)
            interactions['dst'] = interactions['dst'].apply(get_username)

        return interactions

    def run_pipeline(self, config):
        self.is_running = True
        self.timings = {}
        total_start = time.time()

        try:
            outdir = config['workdir']
            os.makedirs(outdir, exist_ok=True)

            #  1. Load data 
            self.log("\n[STEP 1] Loading data...")
            start = time.time()

            data_path = self.uploaded_files.get('dataset')
            if not data_path or not os.path.exists(data_path):
                self.log("[ERROR] Dataset CSV not found"); return

            df = pd.read_csv(data_path)
            self.log(f"  Rows: {len(df):,}")
            self.log(f"  Columns: {list(df.columns)}")

            # Validate required columns
            required = {'from_id', config['text_col']}
            missing = required - set(df.columns)
            if missing:
                self.log(f"[ERROR] CSV missing required columns: {missing}"); return

            # Build username_map: user_id -> username
            username_map = {}
            if 'from_username' in df.columns:
                _u = df[['from_id', 'from_username']].dropna(subset=['from_username']).drop_duplicates('from_id')
                username_map.update(dict(zip(_u['from_id'].apply(self._normalize_id), _u['from_username'])))
            # Add usernames from mentioned column
            if 'mentioned' in df.columns:
                for _, row in df[df['mentioned'].notna()].iterrows():
                    for item in self._parse_mentioned(row['mentioned']):
                        uid = self._normalize_id(item['user_id'])
                        if uid and uid not in username_map and item.get('username'):
                            username_map[uid] = item['username']
            self.log(f"  Username map: {len(username_map):,} users")

            def get_username(uid):
                nid = self._normalize_id(uid)
                return username_map.get(nid, f"@{nid}")

            elapsed = time.time() - start
            self.timings["Load data"] = elapsed
            self.log(f"[DONE] Data loaded. (Time: {elapsed:.2f}s)")

            #  2. Sentiment analysis (fine-tuned model)
            self.log("\n[STEP 2] Fine-tuned sentiment inference...")
            start = time.time()

            proba_cols = tuple(config['proba_cols'].split(','))
            model_path = config['finetuned_model_path']
            text_col = config['text_col']
            batch_size = int(config['batch_size'])
            max_length = int(config['max_length'])

            self.log(f"  Model : {model_path}")
            self.log(f"  Texts : {len(df):,}  (col={text_col}, batch={batch_size}, max_len={max_length})")

            #  Cache: skip inference if results already exist 
            dataset_name = os.path.splitext(os.path.basename(data_path))[0]
            cache_path = os.path.join(outdir, f"{dataset_name}_sentiment_cache.csv")

            if os.path.exists(cache_path):
                self.log(f"  Cache ditemukan: {cache_path}")
                df_sent = pd.read_csv(cache_path)
                self.log(f"  Loaded {len(df_sent):,} cached rows.")
            else:
                self.log(f"  No cache. Running inference (pertama kali, akan di-cache)...")
                try:
                    from transformers import AutoTokenizer, AutoModelForSequenceClassification
                    import torch
                except ImportError:
                    self.log("[ERROR] pip install transformers torch"); return

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.log(f"  Device: {device}")

                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
                model.eval()

                id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
                self.log(f"  id2label: {id2label}")

                text_list = [str(t)[:512] for t in df[text_col].fillna('')]
                total_batches = math.ceil(len(text_list) / batch_size)
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
                        label = id2label.get(int(preds[j]), 'neutral')
                        all_labels.append(label)
                        prob_map = {id2label.get(idx, 'neutral'): float(p) for idx, p in enumerate(probs[j])}
                        all_probs.append(prob_map)

                    done = min(bi + batch_size, len(text_list))
                    batch_num = (bi // batch_size) + 1
                    if batch_num % 50 == 0 or done == len(text_list):
                        self.log(f"  Progress: {done:,}/{len(text_list):,} ({100*done/len(text_list):.1f}%)")

                df_sent = pd.DataFrame({
                    'indobert_label': all_labels,
                    'indobert_neg': [p.get('negative', 0.0) for p in all_probs],
                    'indobert_neu': [p.get('neutral', 0.0) for p in all_probs],
                    'indobert_pos': [p.get('positive', 0.0) for p in all_probs],
                })
                df_sent.to_csv(cache_path, index=False)
                self.log(f"  Cached to: {cache_path}")

                # Free GPU memory
                del model, tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Map to proba_cols
            self.log(f"  df_sent columns: {list(df_sent.columns)}")
            self.log(f"  df_sent shape: {df_sent.shape}")
            col_neg, col_neu, col_pos = proba_cols

            sent_col_map = {
                'negative': col_neg, 'neg': col_neg, 'indobert_neg': col_neg,
                'neutral': col_neu, 'neu': col_neu, 'indobert_neu': col_neu,
                'positive': col_pos, 'pos': col_pos, 'indobert_pos': col_pos,
            }
            label_col = next((c for c in df_sent.columns if 'label' in c.lower()), None)

            for src_col in df_sent.columns:
                if src_col in sent_col_map:
                    df[sent_col_map[src_col]] = df_sent[src_col].values
            if label_col:
                df['sentiment_label'] = df_sent[label_col].values

            if col_neg not in df.columns or col_pos not in df.columns:
                self.log(f"[ERROR] Sentiment columns not mapped. df_sent has: {list(df_sent.columns)}")
                return

            elapsed = time.time() - start
            self.timings["Sentiment inference"] = elapsed
            self.log(f"  Label distribution: {df_sent['indobert_label'].value_counts().to_dict()}")
            self.log(f"[DONE] Sentiment inference complete. (Time: {elapsed:.2f}s)")

            #  3. Build interactions & graph ─
            self.log("\n[STEP 3] Building interactions & graph...")
            start = time.time()

            df_interactions = self._build_interactions(df, get_username=get_username)
            if len(df_interactions) == 0:
                self.log("[ERROR] No interactions found. Check reply_to_user_id / retweet_from_user_id / mentioned columns."); return

            G, df_nodes = build_graph(df_interactions, directed=True)
            self.log(f"  Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}")

            elapsed = time.time() - start
            self.timings["Build graph"] = elapsed
            self.log(f"[DONE] Graph built. (Time: {elapsed:.2f}s)")

            #  4. Estimate user leaning
            self.log("\n[STEP 4] Estimating user leaning...")
            start = time.time()

            topic_polarity = config['topic_polarity']

            # Prepare content with user column = username (converted from from_id)
            df_content = df[['from_id'] + list(proba_cols)].copy()
            df_content['user'] = df_content['from_id'].apply(get_username)
            df_content = df_content.drop(columns=['from_id'])
            df_content = df_content.dropna(subset=['user'])

            df_users = estimate_user_leaning(
                df_content,
                user_col='user',
                proba_cols=proba_cols,
                topic_polarity=topic_polarity,
            )
            self.log(f"  Users with leaning: {len(df_users)}")

            elapsed = time.time() - start
            self.timings["User leaning"] = elapsed
            self.log(f"[DONE] User leaning estimated. (Time: {elapsed:.2f}s)")

            #  5. Calibration (optional)
            if config['calibration_enabled']:
                self.log("\n[STEP 5] Calibrating probabilities...")
                start = time.time()
                cal_method = config['calibration_method']

                df_users = calibrate_probabilities(
                    df_users,
                    method=cal_method,
                    proba_cols=("p_neg_mean", "p_neu_mean", "p_pos_mean"),
                )
                self.log(f"  Calibration method: {cal_method}")

                elapsed = time.time() - start
                self.timings["Calibration"] = elapsed
                self.log(f"[DONE] Calibration complete. (Time: {elapsed:.2f}s)")
            else:
                self.log("\n[STEP 5] Calibration skipped.")

            #  6. Community detection (Leiden + Infomap parallel) 
            self.log("\n[STEP 6] Detecting communities (Leiden + Infomap)...")
            start = time.time()

            resolution = config['resolution']
            if config['auto_resolution']:
                resolution = max(0.5, math.log10(G.number_of_nodes() + 1))
                self.log(f"  Auto resolution: {resolution:.3f}")
            else:
                self.log(f"  Resolution: {resolution:.3f}")

            rs = int(config['random_state'])

            # Run both algorithms
            comms_leiden = detect_communities(G, method='leiden', resolution=resolution, random_state=rs)
            n_leiden = len(set(comms_leiden.values()))
            self.log(f"  Leiden  : {n_leiden} communities")

            comms_infomap = detect_communities(G, method='infomap', random_state=rs)
            n_infomap = len(set(comms_infomap.values()))
            self.log(f"  Infomap : {n_infomap} communities")

            elapsed = time.time() - start
            self.timings["Community detection"] = elapsed
            self.log(f"[DONE] Community detection complete. (Time: {elapsed:.2f}s)")

            #  6b. Community quality metrics ─
            self.log("\n[STEP 6b] Validating community quality...")
            start_q = time.time()

            from networkx.algorithms.community.quality import modularity as nx_modularity, partition_quality

            G_undirected = G.to_undirected() if G.is_directed() else G

            quality_metrics = {}
            for label, comms in [('Leiden', comms_leiden), ('Infomap', comms_infomap)]:
                # Convert dict to list-of-sets partition
                from collections import defaultdict as _dd
                _comm_sets = _dd(set)
                for user, cid in comms.items():
                    if user in G_undirected:
                        _comm_sets[cid].add(user)
                partition = list(_comm_sets.values())

                # Modularity
                mod = nx_modularity(G_undirected, partition, weight='weight')
                # Coverage & Performance
                coverage, performance = partition_quality(G_undirected, partition)
                # Conductance (mean per community)
                conductances = []
                for comm_set in partition:
                    if len(comm_set) < 2:
                        continue
                    internal = 0
                    boundary = 0
                    for u in comm_set:
                        for v in G_undirected.neighbors(u):
                            if v in comm_set:
                                internal += 1
                            else:
                                boundary += 1
                    denom = 2 * internal + boundary
                    if denom > 0:
                        conductances.append(boundary / denom)
                mean_cond = float(np.mean(conductances)) if conductances else 0.0

                # NMI between Leiden and Infomap
                nmi = None  # will compute after both are done

                quality_metrics[label] = {
                    'modularity': mod,
                    'coverage': coverage,
                    'performance': performance,
                    'mean_conductance': mean_cond,
                }
                self.log(f"\n  [{label}] Quality Metrics:")
                self.log(f"    Modularity     : {mod:.4f}  (>0.3 = meaningful structure)")
                self.log(f"    Coverage       : {coverage:.4f}  (fraction of intra-community edges)")
                self.log(f"    Performance    : {performance:.4f}  (correctly placed node pairs)")
                self.log(f"    Conductance    : {mean_cond:.4f}  (lower = better separated, <0.5 good)")

            # NMI between Leiden and Infomap
            try:
                from sklearn.metrics import normalized_mutual_info_score
                common_users = sorted(set(comms_leiden.keys()) & set(comms_infomap.keys()))
                if len(common_users) > 10:
                    labels_l = [comms_leiden[u] for u in common_users]
                    labels_i = [comms_infomap[u] for u in common_users]
                    nmi_score = normalized_mutual_info_score(labels_l, labels_i)
                    quality_metrics['nmi'] = nmi_score
                    self.log(f"\n  [NMI Leiden↔Infomap]: {nmi_score:.4f}  (agreement between algorithms)")
                    if nmi_score > 0.5:
                        self.log(f"    -> Kedua algoritma menemukan struktur komunitas yang serupa (NMI > 0.5)")
                    else:
                        self.log(f"    -> Perbedaan signifikan dalam partisi (NMI <= 0.5)")
            except ImportError:
                self.log("  WARN: sklearn not available, NMI skipped")

            # Reliability verdict
            self.log("\n  [VERDICT]")
            for label in ['Leiden', 'Infomap']:
                qm = quality_metrics[label]
                reliable = qm['modularity'] > 0.3 and qm['coverage'] > 0.3 and qm['mean_conductance'] < 0.5
                status = "RELIABLE" if reliable else "PERLU DICEK"
                self.log(f"    {label}: {status} (mod={qm['modularity']:.3f}, cov={qm['coverage']:.3f}, cond={qm['mean_conductance']:.3f})")

            elapsed_q = time.time() - start_q
            self.timings["Quality metrics"] = elapsed_q
            self.log(f"[DONE] Quality validation complete. (Time: {elapsed_q:.2f}s)")

            #  7. Compute ECR 2.0 for both
            self.log("\n[STEP 7] Computing ECR 2.0 (Leiden + Infomap)...")
            start = time.time()

            results = {}
            for label, comms in [('Leiden', comms_leiden), ('Infomap', comms_infomap)]:
                ecr2 = compute_ecr2(G, df_users, comms)
                results[label] = {
                    'ecr2': ecr2, 'communities': comms,
                    'n': len(set(comms.values())),
                    'quality': quality_metrics[label],
                }
                self.log(f"\n  [{label}]")
                self.log(f"    Intra agreement : {ecr2.intra:.4f}")
                self.log(f"    Inter agreement : {ecr2.inter:.4f}")
                self.log(f"    ECR 2.0 ratio   : {ecr2.ratio:.4f}")
                self.log(f"    Homophily r     : {ecr2.homophily_r:.4f}")

            elapsed = time.time() - start
            self.timings["ECR 2.0"] = elapsed
            self.log(f"\n[DONE] ECR 2.0 computed. (Time: {elapsed:.2f}s)")

            #  8. ECR Threshold & Classification for both 
            self.log("\n[STEP 8] Estimating ECR threshold (Leiden + Infomap)...")
            start = time.time()

            n_threshold_samples = int(config['n_threshold_samples'])
            for label in results:
                comms = results[label]['communities']
                threshold = estimate_ecr_threshold(G, df_users, comms, n_samples=n_threshold_samples, seed=rs)
                classification = classify_echo_chamber(results[label]['ecr2'].ratio, threshold)
                results[label]['threshold'] = threshold
                results[label]['classification'] = classification
                self.log(f"  [{label}] Threshold: {threshold:.4f} -> {classification}")

            elapsed = time.time() - start
            self.timings["ECR threshold"] = elapsed
            self.log(f"[DONE] Threshold estimated. (Time: {elapsed:.2f}s)")

            #  8b. ECR Statistical Validation (Bootstrap CI + Permutation Test) ─
            self.log("\n[STEP 8b] ECR Statistical Validation...")
            start = time.time()

            n_boot = 100
            n_perm = 100
            rng = np.random.RandomState(rs)

            for label in results:
                comms = results[label]['communities']
                ecr_observed = results[label]['ecr2'].ratio

                # ── Bootstrap 95% CI ──
                boot_ratios = []
                users_in_graph = [u for u in df_users['user'] if u in G]
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

                # ── Permutation Test ──
                perm_ratios = []
                lean_vals = df_users.set_index('user')['lean_scalar'].to_dict()
                user_list = list(lean_vals.keys())
                lean_array = np.array([lean_vals[u] for u in user_list])
                for _ in range(n_perm):
                    shuffled = rng.permutation(lean_array)
                    df_perm = df_users.copy()
                    perm_map = dict(zip(user_list, shuffled))
                    df_perm['lean_scalar'] = df_perm['user'].map(perm_map)
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

                results[label]['bootstrap_ci'] = (ci_lo, ci_hi)
                results[label]['p_value'] = p_value

                threshold = results[label]['threshold']
                self.log(f"\n  [{label}] Statistical Validation:")
                self.log(f"    ECR 2.0 ratio      : {ecr_observed:.4f}")
                self.log(f"    Bootstrap 95% CI   : [{ci_lo:.4f}, {ci_hi:.4f}]")
                self.log(f"    Permutation p-value: {p_value:.4f} {'(significant)' if p_value < 0.05 else '(not significant)'}")

                # Verdict
                if ecr_observed < threshold and p_value < 0.05 and ci_hi < threshold:
                    verdict = "ECHO CHAMBER DETECTED (strong evidence)"
                elif ecr_observed < threshold and p_value < 0.05:
                    verdict = "ECHO CHAMBER LIKELY (ECR < threshold, p < 0.05)"
                elif ecr_observed < threshold:
                    verdict = "POSSIBLE ECHO CHAMBER (ECR < threshold, but p >= 0.05)"
                else:
                    verdict = "NO ECHO CHAMBER (ECR >= threshold)"
                results[label]['verdict'] = verdict
                self.log(f"    -> {verdict}")

            elapsed = time.time() - start
            self.timings["ECR validation"] = elapsed
            self.log(f"[DONE] ECR validation complete. (Time: {elapsed:.2f}s)")

            #  9. Homophily
            self.log("\n[STEP 9] Computing homophily...")
            start = time.time()

            homo = compute_homophily(G, df_users)
            self.log(f"  Pearson r: {homo['pearson_r']:.4f}")

            elapsed = time.time() - start
            self.timings["Homophily"] = elapsed
            self.log(f"[DONE] Homophily computed. (Time: {elapsed:.2f}s)")

            #  10. Diffusion bias 
            if config['diffusion_enabled']:
                self.log("\n[STEP 10] Simulating diffusion bias (IC)...")
                start = time.time()

                ic_df = simulate_diffusion_bias(
                    G, df_users,
                    p_activate=float(config['p_activate']),
                    n_runs=int(config['n_runs']),
                    seed=rs,
                )
                ic_summary = summarize_diffusion_bias(ic_df)
                self.log(f"  Seeds simulated: {len(ic_df)}")

                elapsed = time.time() - start
                self.timings["Diffusion bias"] = elapsed
                self.log(f"[DONE] Diffusion simulation complete. (Time: {elapsed:.2f}s)")
            else:
                ic_df = None
                ic_summary = None
                self.log("\n[STEP 10] Diffusion simulation skipped.")

            #  11. Community classification (both) ─
            self.log("\n[STEP 11] Classifying communities...")
            start = time.time()

            for label in results:
                comms = results[label]['communities']
                df_comm = classify_communities_detailed(G, df_users, comms)
                results[label]['df_comm'] = df_comm
                self.log(f"  [{label}] {len(df_comm)} communities classified")

            elapsed = time.time() - start
            self.timings["Community classification"] = elapsed
            self.log(f"[DONE] Community classification complete. (Time: {elapsed:.2f}s)")

            #  12. Diffusion bias metrics (both) ─
            if config['diffusion_enabled'] and ic_df is not None:
                self.log("\n[STEP 12] Computing diffusion bias metrics...")
                start = time.time()

                for label in results:
                    comms = results[label]['communities']
                    try:
                        dm = compute_diffusion_bias_metrics(ic_df, comms, df_users)
                        results[label]['diff_metrics'] = dm
                        gm = dm['global_metrics']
                        self.log(f"  [{label}] slope={gm['slope']:.4f}, R2={gm['r_squared']:.4f}, "
                                 f"bias_ratio={gm['bias_ratio']:.4f}, Cohen_d={gm['cohens_d']:.4f}")
                    except Exception as e:
                        self.log(f"  [{label}] WARN: {e}")
                        results[label]['diff_metrics'] = None

                elapsed = time.time() - start
                self.timings["Diffusion metrics"] = elapsed
                self.log(f"[DONE] Diffusion metrics computed. (Time: {elapsed:.2f}s)")

            #  13. Save outputs 
            self.log("\n[STEP 13] Saving outputs...")
            start = time.time()

            # Users with leaning (shared)
            users_path = os.path.join(outdir, "users_leaning.csv")
            df_users.to_csv(users_path, index=False)
            self.log(f"  Saved: {users_path}")

            # Per-algorithm outputs
            for label in results:
                r = results[label]
                prefix = label.lower()
                sub = os.path.join(outdir, prefix)
                os.makedirs(sub, exist_ok=True)

                # Communities CSV
                comm_df = pd.DataFrame([{"user": u, "community": c} for u, c in r['communities'].items()])
                comm_df.to_csv(os.path.join(sub, "communities.csv"), index=False)

                # Community classification
                r['df_comm'].to_csv(os.path.join(sub, "community_classification.csv"), index=False)

                # ECR metrics
                ecr2 = r['ecr2']
                with open(os.path.join(sub, "ecr_metrics.txt"), 'w', encoding='utf-8') as f:
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
                    qm = r.get('quality', {})
                    if qm:
                        f.write(f"\n--- Community Quality Metrics ---\n")
                        f.write(f"Modularity                 : {qm['modularity']:.6f}\n")
                        f.write(f"Coverage                   : {qm['coverage']:.6f}\n")
                        f.write(f"Performance                : {qm['performance']:.6f}\n")
                        f.write(f"Mean Conductance           : {qm['mean_conductance']:.6f}\n")
                    bci = r.get('bootstrap_ci')
                    pv = r.get('p_value')
                    vd = r.get('verdict')
                    if bci is not None:
                        f.write(f"\n--- Statistical Validation ---\n")
                        f.write(f"Bootstrap 95% CI           : [{bci[0]:.6f}, {bci[1]:.6f}]\n")
                        f.write(f"Permutation p-value        : {pv:.6f}\n")
                        f.write(f"Verdict                    : {vd}\n")

                # Diffusion bias metrics
                dm = r.get('diff_metrics')
                if dm is not None:
                    dm['community_metrics'].to_csv(os.path.join(sub, "diffusion_bias_metrics.csv"), index=False)

                self.log(f"  Saved: {sub}/")

            # Diffusion results (shared, not per-algorithm)
            if ic_df is not None:
                ic_df.to_csv(os.path.join(outdir, "diffusion_ic.csv"), index=False)
                ic_summary.to_csv(os.path.join(outdir, "diffusion_summary.csv"), index=False)
                self.log(f"  Saved: diffusion_ic.csv, diffusion_summary.csv")

            # Combined comparison summary
            summary_path = os.path.join(outdir, "comparison_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"{'='*60}\n")
                f.write(f"  ECR 2.0 — Leiden vs Infomap Comparison\n")
                f.write(f"{'='*60}\n\n")
                f.write(f"{'Metric':<30} {'Leiden':>12} {'Infomap':>12}\n")
                f.write(f"{'-'*54}\n")
                f.write(f"{'Communities':<30} {results['Leiden']['n']:>12} {results['Infomap']['n']:>12}\n")
                f.write(f"{'Intra agreement':<30} {results['Leiden']['ecr2'].intra:>12.4f} {results['Infomap']['ecr2'].intra:>12.4f}\n")
                f.write(f"{'Inter agreement':<30} {results['Leiden']['ecr2'].inter:>12.4f} {results['Infomap']['ecr2'].inter:>12.4f}\n")
                f.write(f"{'ECR 2.0 ratio':<30} {results['Leiden']['ecr2'].ratio:>12.4f} {results['Infomap']['ecr2'].ratio:>12.4f}\n")
                f.write(f"{'Threshold':<30} {results['Leiden']['threshold']:>12.4f} {results['Infomap']['threshold']:>12.4f}\n")
                f.write(f"{'Classification':<30} {results['Leiden']['classification']:>12} {results['Infomap']['classification']:>12}\n")
                # Quality metrics
                f.write(f"\n{'='*60}\n")
                f.write(f"  Community Quality Validation\n")
                f.write(f"{'='*60}\n\n")
                f.write(f"{'Metric':<30} {'Leiden':>12} {'Infomap':>12}\n")
                f.write(f"{'-'*54}\n")
                ql = results['Leiden'].get('quality', {})
                qi = results['Infomap'].get('quality', {})
                f.write(f"{'Modularity':<30} {ql.get('modularity',0):>12.4f} {qi.get('modularity',0):>12.4f}\n")
                f.write(f"{'Coverage':<30} {ql.get('coverage',0):>12.4f} {qi.get('coverage',0):>12.4f}\n")
                f.write(f"{'Performance':<30} {ql.get('performance',0):>12.4f} {qi.get('performance',0):>12.4f}\n")
                f.write(f"{'Conductance (mean)':<30} {ql.get('mean_conductance',0):>12.4f} {qi.get('mean_conductance',0):>12.4f}\n")
                nmi_val = quality_metrics.get('nmi')
                if nmi_val is not None:
                    f.write(f"\nNMI (Leiden <-> Infomap) : {nmi_val:.4f}\n")

                # Statistical Validation
                f.write(f"\n{'='*60}\n")
                f.write(f"  ECR Statistical Validation\n")
                f.write(f"{'='*60}\n\n")
                f.write(f"{'Metric':<30} {'Leiden':>12} {'Infomap':>12}\n")
                f.write(f"{'-'*54}\n")
                bl = results['Leiden'].get('bootstrap_ci', (0, 0))
                bi = results['Infomap'].get('bootstrap_ci', (0, 0))
                f.write(f"{'Bootstrap CI low':<30} {bl[0]:>12.4f} {bi[0]:>12.4f}\n")
                f.write(f"{'Bootstrap CI high':<30} {bl[1]:>12.4f} {bi[1]:>12.4f}\n")
                f.write(f"{'Permutation p-value':<30} {results['Leiden'].get('p_value',1):>12.4f} {results['Infomap'].get('p_value',1):>12.4f}\n")
                f.write(f"\nVerdict Leiden  : {results['Leiden'].get('verdict', 'N/A')}\n")
                f.write(f"Verdict Infomap: {results['Infomap'].get('verdict', 'N/A')}\n")

                f.write(f"\nHomophily (Pearson r): {homo['pearson_r']:.4f}\n")
                f.write(f"Topic polarity       : {topic_polarity}\n")
                f.write(f"Resolution (Leiden)  : {resolution:.3f}\n")
            self.log(f"  Saved: {summary_path}")

            elapsed = time.time() - start
            self.timings["Save outputs"] = elapsed
            self.log(f"[DONE] Outputs saved. (Time: {elapsed:.2f}s)")

            #  14. Generate visualizations ─
            self.log("\n[STEP 14] Generating visualizations...")
            start = time.time()

            viz_dir = os.path.join(outdir, "visualizations")
            self.viz_paths = generate_all_visualizations(
                G, df_users, results, homo, ic_df, viz_dir,
                df=df, df_sent=df_sent, log_fn=self.log,
            )

            elapsed = time.time() - start
            self.timings["Visualizations"] = elapsed
            self.log(f"[DONE] {len(self.viz_paths)} visualizations saved to {viz_dir}/ (Time: {elapsed:.2f}s)")

            #  Summary ─
            total_elapsed = time.time() - total_start
            self.log("\n" + "=" * 60)
            self.log("[PIPELINE COMPLETE]")
            self.log(f"Output directory: {os.path.abspath(outdir)}")
            self.log(f"\n[PIPELINE TIMINGS]")
            for comp, t in self.timings.items():
                self.log(f"  {comp}: {t:.2f}s")
            self.log(f"  Total: {total_elapsed:.2f}s")
            self.log("=" * 60)

        except Exception as e:
            self.log(f"\n[ERROR] Pipeline failed:\n{traceback.format_exc()}")
        finally:
            self.is_running = False


def create_gui():
    pipeline = LeidenPipelineGUI()

    # Resolve absolute path to finetuned model relative to this script
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _default_finetuned = os.path.join(_script_dir, "finetuned_sentiment_cardiff_xlmroberta")

    config = {
        'workdir': 'pipeline_out',
        # Sentiment — fine-tuned model inference
        'finetuned_model_path': _default_finetuned,
        'text_col': 'cleaned_text',
        'batch_size': 64,
        'max_length': 128,
        'proba_cols': 'p_neg,p_neu,p_pos',
        # User leaning
        'topic_polarity': 'negative_is_pro',
        # Calibration
        'calibration_enabled': False,
        'calibration_method': 'isotonic',
        # Community detection (both Leiden + Infomap always run)
        'resolution': 1.0,
        'auto_resolution': True,
        # ECR threshold
        'n_threshold_samples': 100,
        # Diffusion
        'diffusion_enabled': True,
        'p_activate': 0.05,
        'n_runs': 20,
        # Misc
        'random_state': 42,
    }

    def info_label(text, tooltip):
        """Render a label with an (i) info icon tooltip."""
        with ui.row().classes('items-center gap-1'):
            ui.label(text).classes('text-sm font-medium')
            with ui.icon('info', size='xs').classes('text-blue-400 cursor-help'):
                ui.tooltip(tooltip).classes('max-w-xs')

    @ui.page('/')
    def main_page():

        ui.add_head_html('''
            <style>
                .nicegui-content {
                    padding: 2rem;
                    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
                    min-height: 100vh;
                }
                .q-card {
                    border-radius: 12px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.15);
                    backdrop-filter: blur(10px);
                    background: rgba(255,255,255,0.97);
                }
                .q-btn {
                    border-radius: 8px;
                    font-weight: 600;
                }
                .q-input, .q-select {
                    border-radius: 6px;
                }
            </style>
        ''')

        with ui.column().classes('w-[920px] mx-auto space-y-6'):

            #  Header 
            ui.markdown('# ECR 2.0 Pipeline — Leiden').classes('text-center mb-2 text-white')
            ui.markdown('*Echo Chamber Ratio 2.0 with Leiden Community Detection*').classes('text-center mb-6 text-gray-300')

            #  Dataset Input ─
            with ui.card().classes('w-full p-6'):
                ui.label('Dataset Input').classes('text-xl font-semibold mb-4')

                info_label(
                    'Dataset CSV (total_data_cleaned) *',
                    'File CSV hasil crawling Twitter/X. Harus mengandung kolom: from_id, cleaned_text, '
                    'dan minimal satu dari: reply_to_user_id, retweet_from_user_id, mentioned. '
                    'Kolom-kolom ini digunakan untuk membangun graf interaksi antar user.',
                )
                ui.label('Required: from_id, cleaned_text, + reply_to_user_id / retweet_from_user_id / mentioned').classes('text-xs text-gray-500 mb-2')

                upload_label = ui.label('Belum ada file diupload').classes('text-sm text-gray-500 mt-2')

                async def on_upload(e):
                    try:
                        f = e.file
                        uploads_dir = 'uploads'
                        os.makedirs(uploads_dir, exist_ok=True)
                        upload_path = os.path.join(uploads_dir, f.name)
                        data = await f.read()
                        with open(upload_path, 'wb') as out:
                            out.write(data)
                        pipeline.uploaded_files['dataset'] = os.path.abspath(upload_path)
                        upload_label.set_text(f'Uploaded: {f.name} ({len(data):,} bytes)')
                        upload_label.classes('text-green-600', remove='text-gray-500')
                        ui.notify(f'Dataset uploaded: {f.name}', type='positive')
                    except Exception as ex:
                        upload_label.set_text(f'Upload gagal: {ex}')
                        upload_label.classes('text-red-600', remove='text-gray-500')
                        ui.notify(f'Upload gagal: {ex}', type='negative')

                ui.upload(
                    on_upload=on_upload,
                    auto_upload=True,
                    label='Drag & drop CSV disini atau klik untuk browse',
                ).classes('w-full')

            #  General Settings 
            with ui.card().classes('w-full p-6'):
                ui.label('General Settings').classes('text-xl font-semibold mb-4')

                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        info_label('Output Directory',
                                   'Folder tempat semua hasil pipeline disimpan (CSV, TXT). '
                                   'Jika folder belum ada, akan dibuat otomatis.')
                        workdir_input = ui.input(value=config['workdir']).classes('w-full')
                        workdir_input.bind_value(config, 'workdir')

                    with ui.column().classes('flex-1'):
                        info_label('Random State',
                                   'Seed untuk reproducibility. Mengubah nilai ini akan menghasilkan '
                                   'partisi komunitas, simulasi difusi, dan threshold ECR yang berbeda. '
                                   'Gunakan angka yang sama untuk hasil yang konsisten.')
                        rs_input = ui.number(value=config['random_state'], min=0).classes('w-full')
                        rs_input.bind_value(config, 'random_state')

            #  Sentiment Configuration ─
            with ui.card().classes('w-full p-6'):
                ui.label('Sentiment Configuration').classes('text-xl font-semibold mb-4')
                ui.label(
                    'Menggunakan fine-tuned XLM-RoBERTa model dari folder finetuned_sentiment_cardiff_xlmroberta/. '
                    'Hasil inferensi di-cache otomatis agar tidak perlu diulang.'
                ).classes('text-xs text-gray-500 mb-2')

                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        info_label('Model Path',
                                   'Path ke folder fine-tuned model (berisi config.json, model.safetensors, tokenizer). '
                                   'Default mengarah ke folder finetuned_sentiment_cardiff_xlmroberta/ di project ini.')
                        ft_path_input = ui.input(value=config['finetuned_model_path']).classes('w-full')
                        ft_path_input.bind_value(config, 'finetuned_model_path')

                    with ui.column().classes('flex-1'):
                        info_label('Text Column',
                                   'Nama kolom di CSV yang berisi teks untuk dianalisis sentimen. '
                                   'Biasanya "cleaned_text" untuk teks yang sudah dibersihkan.')
                        text_col_input = ui.input(value=config['text_col']).classes('w-full')
                        text_col_input.bind_value(config, 'text_col')

                with ui.row().classes('w-full gap-4 mt-2'):
                    with ui.column().classes('w-40'):
                        info_label('Batch Size',
                                   'Jumlah teks diproses sekaligus. Lebih besar = lebih cepat tapi butuh lebih banyak RAM/VRAM. '
                                   'Kurangi jika out of memory.')
                        bs_input = ui.number(value=config['batch_size'], min=1, max=512).classes('w-full')
                        bs_input.bind_value(config, 'batch_size')

                    with ui.column().classes('w-40'):
                        info_label('Max Length',
                                   'Panjang maksimum token per teks. Teks lebih panjang dipotong. '
                                   'Default 128 cukup untuk tweet.')
                        ml_input = ui.number(value=config['max_length'], min=32, max=512).classes('w-full')
                        ml_input.bind_value(config, 'max_length')

                    with ui.column().classes('flex-1'):
                        info_label('Probability Columns (neg,neu,pos)',
                                   'Nama 3 kolom output probabilitas sentimen (dipisah koma).')
                        proba_input = ui.input(value=config['proba_cols']).classes('w-full')
                        proba_input.bind_value(config, 'proba_cols')

            #  User Leaning 
            with ui.card().classes('w-full p-6'):
                ui.label('User Leaning').classes('text-xl font-semibold mb-4')

                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        info_label('Topic Polarity',
                                   'Menentukan bagaimana sentimen dipetakan ke posisi user. '
                                   '"negative_is_pro": sentimen negatif berarti PRO terhadap topik '
                                   '(misal: topik "Indonesia Gelap" — komentar negatif = mendukung isu). '
                                   '"positive_is_pro": sentimen positif = PRO topik (misal: topik vaksinasi).')
                        polarity_select = ui.select(
                            ['negative_is_pro', 'positive_is_pro'],
                            value=config['topic_polarity'],
                        ).classes('w-full')
                        polarity_select.bind_value(config, 'topic_polarity')

                    with ui.column().classes('flex-1'):
                        info_label('Calibration',
                                   'Kalibrasi probabilitas sentimen agar lebih akurat. '
                                   'Isotonic: non-parametrik, cocok jika data cukup besar. '
                                   'Temperature: scaling sederhana pada logits. '
                                   'Aktifkan jika model cenderung overconfident atau underconfident.')
                        cal_switch = ui.switch('Enable Calibration', value=config['calibration_enabled'])
                        cal_switch.bind_value(config, 'calibration_enabled')

                # Conditional: calibration method
                cal_container = ui.column().classes('w-full mt-2')

                def update_calibration_settings():
                    cal_container.clear()
                    with cal_container:
                        if config['calibration_enabled']:
                            with ui.row().classes('w-full gap-4'):
                                with ui.column().classes('flex-1'):
                                    info_label('Calibration Method',
                                               '"isotonic": regresi isotonic per kelas — fleksibel, non-parametrik. '
                                               '"temperature": scaling tunggal pada logits — lebih stabil untuk data kecil.')
                                    cal_method_select = ui.select(
                                        ['isotonic', 'temperature'],
                                        value=config['calibration_method'],
                                    ).classes('w-full')
                                    cal_method_select.bind_value(config, 'calibration_method')

                cal_switch.on('update:model-value', lambda: update_calibration_settings())
                update_calibration_settings()

            #  Community Detection ─
            with ui.card().classes('w-full p-6'):
                ui.label('Community Detection').classes('text-xl font-semibold mb-4')
                ui.label('Kedua algoritma (Leiden + Infomap) dijalankan secara parallel dan hasil dibandingkan.').classes('text-xs text-gray-500 mb-2')

                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        info_label('Resolution (Leiden)',
                                   'Mengontrol granularitas komunitas. Nilai lebih tinggi = lebih banyak komunitas kecil. '
                                   'Nilai lebih rendah = lebih sedikit komunitas besar. '
                                   'Mode "Auto" menghitung resolution = max(0.5, log10(jumlah_node + 1)).')
                        auto_res_switch = ui.switch('Auto (log10-based)', value=config['auto_resolution'])
                        auto_res_switch.bind_value(config, 'auto_resolution')

                # Conditional: manual resolution
                res_container = ui.column().classes('w-full mt-2')

                def update_resolution_settings():
                    res_container.clear()
                    with res_container:
                        if not config['auto_resolution']:
                            info_label('Manual Resolution',
                                       'Atur resolution secara manual. Rentang umum: 0.5-5.0. '
                                       'Coba beberapa nilai untuk melihat dampaknya pada jumlah komunitas.')
                            res_input = ui.number(value=config['resolution'], min=0.1, max=10.0, step=0.1).classes('w-48')
                            res_input.bind_value(config, 'resolution')

                auto_res_switch.on('update:model-value', lambda: update_resolution_settings())
                update_resolution_settings()

            #  Diffusion & Threshold ─
            with ui.card().classes('w-full p-6'):
                ui.label('Diffusion Simulation & ECR Threshold').classes('text-xl font-semibold mb-4')

                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('flex-1'):
                        info_label('Diffusion Simulation',
                                   'Simulasi Independent Cascade (IC) untuk mengukur bias difusi informasi. '
                                   'Mengukur apakah informasi cenderung menyebar ke user dengan leaning serupa. '
                                   'Matikan jika ingin pipeline lebih cepat (step ini paling lama).')
                        diff_switch = ui.switch('Enable Diffusion Simulation', value=config['diffusion_enabled'])
                        diff_switch.bind_value(config, 'diffusion_enabled')

                # Conditional: diffusion parameters
                diff_container = ui.column().classes('w-full mt-2')

                def update_diffusion_settings():
                    diff_container.clear()
                    with diff_container:
                        if config['diffusion_enabled']:
                            with ui.row().classes('w-full gap-4'):
                                with ui.column().classes('flex-1'):
                                    info_label('Activation Probability',
                                               'Probabilitas satu node mengaktifkan tetangganya per edge. '
                                               'Nilai lebih tinggi = informasi menyebar lebih luas. '
                                               'Default 0.05 realistis untuk social media.')
                                    p_input = ui.number(value=config['p_activate'], min=0.001, max=1.0, step=0.01, format='%.3f').classes('w-full')
                                    p_input.bind_value(config, 'p_activate')

                                with ui.column().classes('flex-1'):
                                    info_label('IC Runs per Seed',
                                               'Jumlah simulasi IC per seed node. Lebih banyak = estimasi lebih stabil '
                                               'tapi lebih lama. Default 20 cukup untuk hasil yang reliable.')
                                    nr_input = ui.number(value=config['n_runs'], min=1, max=100).classes('w-full')
                                    nr_input.bind_value(config, 'n_runs')

                                with ui.column().classes('flex-1'):
                                    info_label('Threshold Null Samples',
                                               'Jumlah graf random untuk estimasi threshold ECR. '
                                               'Lebih banyak = threshold lebih akurat tapi lebih lama. '
                                               'Default 100. Kurangi ke 20-30 untuk testing cepat.')
                                    ts_input = ui.number(value=config['n_threshold_samples'], min=10, max=500).classes('w-full')
                                    ts_input.bind_value(config, 'n_threshold_samples')

                diff_switch.on('update:model-value', lambda: update_diffusion_settings())
                update_diffusion_settings()

            _prev_viz_count = [0]  # shared state for viz gallery refresh

            #  Run Button
            with ui.card().classes('w-full p-6 text-center'):
                with ui.row().classes('justify-center gap-4'):
                    run_button = ui.button('Run Pipeline', color='primary').classes('text-lg px-12 py-4')
                    load_button = ui.button('Load Previous Results', color='secondary').classes('text-lg px-8 py-4')

                ui.label(
                    'Untuk demo/presentasi: jalankan pipeline sekali, lalu gunakan "Load Previous Results" '
                    'untuk menampilkan hasil instan tanpa menunggu.'
                ).classes('text-xs text-gray-500 mt-3')

                def validate_and_run():
                    if pipeline.is_running:
                        ui.notify('Pipeline already running!', type='warning')
                        return

                    ds = pipeline.uploaded_files.get('dataset', '')
                    if not ds or not os.path.isfile(ds):
                        ui.notify('Upload dataset CSV terlebih dahulu!', type='negative')
                        return

                    _prev_viz_count[0] = 0
                    pipeline.viz_paths = []
                    log_area.set_value('')
                    ui.notify('Pipeline started!', type='positive')
                    threading.Thread(
                        target=pipeline.run_pipeline,
                        args=(config.copy(),),
                        daemon=True,
                    ).start()

                def load_previous_results():
                    """Load visualizations from a previous pipeline run."""
                    outdir = config['workdir']
                    viz_dir = os.path.join(outdir, 'visualizations')
                    if not os.path.isdir(viz_dir):
                        ui.notify(f'Folder tidak ditemukan: {viz_dir}', type='negative')
                        return

                    from glob import glob
                    pngs = sorted(glob(os.path.join(viz_dir, '*.png')))
                    if not pngs:
                        ui.notify(f'Tidak ada visualisasi di {viz_dir}', type='negative')
                        return

                    pipeline.viz_paths = pngs
                    _prev_viz_count[0] = 0  # trigger gallery refresh

                    # Load comparison_summary.txt if exists
                    summary_path = os.path.join(outdir, 'comparison_summary.txt')
                    if os.path.isfile(summary_path):
                        with open(summary_path, 'r', encoding='utf-8') as f:
                            log_area.set_value(f"[LOADED] Previous results from {outdir}/\n\n{f.read()}")
                    else:
                        log_area.set_value(f"[LOADED] {len(pngs)} visualizations from {viz_dir}/")

                    ui.notify(f'Loaded {len(pngs)} visualizations!', type='positive')

                run_button.on('click', validate_and_run)
                load_button.on('click', load_previous_results)

            #  Status Section 
            with ui.card().classes('w-full p-6'):
                ui.label('Status').classes('text-xl font-semibold mb-4')
                status_container = ui.column().classes('w-full')

            #  Logs Section
            with ui.card().classes('w-full p-6'):
                ui.label('Pipeline Logs').classes('text-xl font-semibold mb-4')
                log_area = ui.textarea().classes('w-full h-96 font-mono text-sm').props('readonly outlined')

            #  Visualizations Gallery ─
            with ui.card().classes('w-full p-6'):
                ui.label('Visualizations').classes('text-xl font-semibold mb-4')
                viz_container = ui.column().classes('w-full')
                with viz_container:
                    ui.label('Visualisasi akan muncul setelah pipeline selesai.').classes('text-gray-500')

        #  Timer-based updates ─
        def update_logs():
            content = ''
            while not pipeline.log_queue.empty():
                content += pipeline.log_queue.get() + '\n'
            if content:
                current_value = log_area.value if log_area.value else ''
                log_area.set_value(current_value + content)

        def update_status():
            status_container.clear()
            with status_container:
                ds_path = pipeline.uploaded_files.get('dataset', '')
                dataset_ok = bool(ds_path) and os.path.isfile(ds_path)

                if pipeline.is_running:
                    ui.label('Running Pipeline...').classes('text-lg text-blue-600 font-medium')
                    run_button.props('loading')
                elif not dataset_ok:
                    ui.label('Upload dataset CSV untuk memulai').classes('text-lg text-orange-600 font-medium')
                    run_button.props(remove='loading')
                else:
                    ui.label('Ready to run').classes('text-lg text-green-600 font-medium')
                    ui.label(f'Dataset: {os.path.basename(ds_path)}').classes('text-sm text-gray-600 mt-1')
                    run_button.props(remove='loading')

        def update_viz_gallery():
            """Refresh the visualization gallery when new images are available."""
            current_count = len(pipeline.viz_paths)
            if current_count > 0 and current_count != _prev_viz_count[0]:
                _prev_viz_count[0] = current_count
                viz_container.clear()
                with viz_container:
                    viz_names = {
                        '01_calibration_impact.png': 'Reliability Diagrams — Kalibrasi',
                        '01b_calibration_per_confidence.png': 'Kalibrasi per Confidence',
                        '02_user_leaning.png': 'Distribusi User Leaning',
                        '03_community_sizes.png': 'Ukuran Komunitas',
                        '04_ecr_results.png': 'ECR 2.0 Results',
                        '05_diffusion_bias.png': 'Diffusion Bias',
                        '06_pro_contra.png': 'Pro vs Contra',
                        '07_shifts.png': 'Pergeseran Sentimen',
                        '08_network.png': 'Network Graph',
                        '09_degrees.png': 'Degree Distribution',
                        '10_sentiment_sample.png': 'Contoh Hasil Sentimen (10 Rows)',
                    }
                    for img_path in pipeline.viz_paths:
                        if os.path.exists(img_path):
                            fname = os.path.basename(img_path)
                            title = viz_names.get(fname, fname)
                            ui.label(title).classes('text-lg font-semibold mt-4 mb-2')
                            ui.image(img_path).classes('w-full rounded-lg shadow')

        def safe_timer_tick():
            try:
                update_logs()
                update_status()
                update_viz_gallery()
            except Exception:
                pass  # client disconnected / deleted

        ui.timer(0.5, safe_timer_tick)

    port = int(os.environ.get('PORT', 8080))
    ui.run(title='ECR 2.0 Pipeline — Leiden', host='0.0.0.0', port=port, dark=False)


if __name__ in {"__main__", "__mp_main__"}:
    create_gui()
