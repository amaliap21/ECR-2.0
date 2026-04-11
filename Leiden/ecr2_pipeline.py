"""
Echo Chamber Ratio 2.0 (ECR 2.0) end-to-end pipeline

This module implements a practical, modular pipeline to compute Echo Chamber
Ratio 2.0 on a directed social graph (e.g., X/Twitter interactions):

Components
----------
1) Graph building from interactions (retweet / reply / mention / quote).
2) User leaning estimation from content or supported content (per-tweet or per-domain labels).
3) Optional probabilistic calibration (temperature scaling, isotonic regression).
4) Community detection using Leiden and/or Infomap (directed or undirected views).
5) Homophily measurement (correlation between user leaning and neighbors' mean leaning).
6) Diffusion bias via Independent Cascade (IC) simulation and influence-set leaning.
7) ECR 2.0 = E[ intra-community agreement ] / E[ inter-community agreement ].
8) Utilities and light visualizations.

Input expectations
------------------
- interactions: pandas.DataFrame with columns at least:
    ['src', 'dst', 'type', 'weight']
  where `type`∈{'retweet','reply','mention','quote'} (arbitrary strings allowed)
  and `weight` is numeric (defaults to 1 if missing).

- content_lean: pandas.DataFrame with one of the following schemas:
    A) per-user soft leaning probabilities (recommended):
       columns: ['user', 'p_neg', 'p_neu', 'p_pos'] OR ['user', 'proba_*']
       → converted to scalar leaning in [-1, +1] via expectation mapping.

    B) per-tweet/URL/domain labels that can be aggregated to user level, e.g.:
       columns: ['user','content_id','lean_label'] with label in {-1, 0, +1} or {'contra','neutral','pro'}

You can mix / adapt by precomputing a per-user leaning table and passing it here.

Install hints (CPU-only ok)
    pip install pandas numpy networkx igraph leidenalg infomap==2.7.1 scikit-learn tqdm deep-translator

If `igraph`/`leidenalg` or `infomap` is not available, the code falls back to
built-ins (e.g., Louvain via `networkx` community when possible), but Leiden/Infomap
is strongly recommended for well-connected communities and flow-based structure.

Example
-------
    from ecr2_pipeline import (
        build_graph, estimate_user_leaning, calibrate_probabilities,
        detect_communities, compute_homophily,
        simulate_diffusion_bias, compute_ecr2, ECR2Result
    )

    G, df_nodes = build_graph(interactions_df)
    df_users = estimate_user_leaning(content_df)
    df_users = calibrate_probabilities(df_users, method='isotonic')  # optional

    part = detect_communities(G, method='leiden', resolution=1.0)
    homo = compute_homophily(G, df_users)
    diff = simulate_diffusion_bias(G, df_users, p_activate=0.05, n_runs=20, seed=42)

    ecr2 = compute_ecr2(G, df_users, part)
    print(ecr2)
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

# Optional deps
try:
    import igraph as ig  # type: ignore
    import leidenalg as la  # type: ignore
    _HAS_LEIDEN = True
except Exception:
    _HAS_LEIDEN = False

try:
    from infomap import Infomap  # type: ignore
    _HAS_INFOMAP = True
except Exception:
    _HAS_INFOMAP = False

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss
    from sklearn.model_selection import StratifiedKFold
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    from scipy.optimize import minimize_scalar
    from scipy import stats as scipy_stats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# 1) Graph construction

def build_graph(
    interactions: pd.DataFrame,
    weight_by: Optional[str] = "weight",
    directed: bool = True,
) -> Tuple[nx.DiGraph | nx.Graph, pd.DataFrame]:
    """Build a (di)graph from interaction dataframe.

    Parameters
    ----------
    interactions : DataFrame with columns: ['src','dst',('weight'),'type']
    weight_by   : column to use as edge weight; if missing, defaults to 1.
    directed    : build DiGraph (True) or Graph (False)

    Returns
    -------
    G, df_nodes
        G is a NetworkX graph, df_nodes lists unique users as DataFrame with column 'user'.
    """
    required = {"src", "dst"}
    if not required.issubset(interactions.columns):
        raise ValueError(f"interactions must contain {required}")

    W = weight_by if (weight_by is not None and weight_by in interactions.columns) else None

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    for _, row in interactions.iterrows():
        s, d = row["src"], row["dst"]
        w = float(row[W]) if W else 1.0
        t = row.get("type", "edge")
        if G.has_edge(s, d):
            G[s][d]["weight"] = G[s][d].get("weight", 0.0) + w
        else:
            G.add_edge(s, d, weight=w, itype=t)

    # Add isolated nodes if present in columns but not in edges
    users = pd.unique(pd.concat([interactions["src"], interactions["dst"]], ignore_index=True))
    for u in users:
        if u not in G:
            G.add_node(u)

    df_nodes = pd.DataFrame({"user": list(G.nodes())})
    return G, df_nodes


# 9) Empirical Threshold Estimation for ECR (0-100 scale) + Classification
# Added: automatic threshold estimation using bootstrapping / null-model
# distribution (degree-preserved randomization). Result is a threshold
# in [0,100] without hardcoding.

import copy


def _randomize_graph_preserve_degree(G: nx.Graph | nx.DiGraph, n_swap: int | None = None, seed: int = 42):
    """Generate a randomized graph with preserved degree sequence.
    For directed graphs uses directed double-edge swaps.

    n_swap defaults to 10 * number_of_edges to ensure thorough randomization.
    """
    H = copy.deepcopy(G)
    if n_swap is None:
        n_swap = max(H.number_of_edges(), 5_000)
    if isinstance(G, nx.DiGraph):
        try:
            nx.algorithms.swap.directed_edge_swap(H, nswap=n_swap, max_tries=n_swap*10, seed=seed)
        except nx.NetworkXAlgorithmError:
            pass
    else:
        try:
            nx.algorithms.swap.double_edge_swap(H, nswap=n_swap, max_tries=n_swap*10, seed=seed)
        except nx.NetworkXAlgorithmError:
            pass
    return H


def estimate_ecr_threshold(
    G: nx.Graph | nx.DiGraph,
    df_users: pd.DataFrame,
    communities: Dict[str,int],
    n_samples: int = 100,
    seed: int = 42,
) -> float:
    """Estimate empirical ECR threshold by generating a null distribution.

    Steps:
    1) randomize graph preserving degree sequence (n_swap = 10 * |E|)
    2) compute ECR on each randomized graph
    3) threshold = mean(null distribution) + 1 * std(null distribution)

    Uses n_samples=100 by default for a more reliable null distribution.
    """
    rng = random.Random(seed)
    vals = []

    for i in range(n_samples):
        H = _randomize_graph_preserve_degree(G, seed=rng.randint(1,10**9))
        res = compute_ecr2(H, df_users, communities)
        if not math.isnan(res.ratio) and res.ratio > 0:
            vals.append(min(1.0, res.ratio))

    if len(vals) < 3:
        return 1.0  # fallback: extremely conservative

    mu = float(np.mean(vals))
    sd = float(np.std(vals))
    threshold = mu + sd
    return float(max(0.0, min(1.0, threshold)))


def classify_echo_chamber(ecr_ratio: float, threshold: float) -> str:
    """Return text label based on ratio vs threshold."""
    if math.isnan(ecr_ratio):
        return "Tidak dapat ditentukan (ECR NaN)"
    val = ecr_ratio
    if val > threshold:
        return "Fenomena Echo Chamber"
    elif val < threshold:
        return "Fenomena Non-Echo Chamber"
    else:
        return "Tidak dapat ditentukan (tepat di threshold)"

# 2) User leaning estimation + calibration

def _labels_to_scalar(y: Iterable) -> np.ndarray:
    mapping = {"contra": -1.0, "neg": -1.0, -1: -1.0, "neutral": 0.0, 0: 0.0, "pos": 1.0, "pro": 1.0, 1: 1.0}
    out = []
    for v in y:
        out.append(mapping.get(v, float(v)))
    return np.asarray(out, dtype=float)


def estimate_user_leaning(
    content: pd.DataFrame,
    user_col: str = "user",
    label_cols: Optional[Tuple[str, ...]] = None,
    label_col: Optional[str] = None,
    proba_cols: Optional[Tuple[str, ...]] = None,
    topic_polarity: str = "negative_is_pro",
) -> pd.DataFrame:
    """Estimate per-user scalar leaning in [-1, +1].

    Provide *either* discrete labels via `label_col` (e.g., {-1,0,+1} or ('contra','neutral','pro'))
    or soft probabilities via `proba_cols` (e.g., ('p_neg','p_neu','p_pos')). If
    you already have per-user rows, pass `content` with one row per user; otherwise
    the function will aggregate by mean.

    Parameters
    ----------
    topic_polarity : str
        'negative_is_pro' (default): sentimen negatif = PRO topik (e.g., "Indonesia Gelap")
            lean_scalar = (+1)*p_neg + 0*p_neu + (-1)*p_pos
        'positive_is_pro': sentimen positif = PRO topik
            lean_scalar = (-1)*p_neg + 0*p_neu + (+1)*p_pos

    Returns DataFrame with columns:
        ['user', 'lean_scalar', 'conf', 'p_neg_mean', 'p_neu_mean', 'p_pos_mean']
    """
    if user_col not in content.columns:
        raise ValueError(f"content must include column '{user_col}'")

    df = content.copy()

    if proba_cols is not None and all(c in df.columns for c in proba_cols):
        # expectation mapping: E[lean] with bins {-1, 0, +1}
        p_neg, p_neu, p_pos = [df[c].astype(float).values for c in proba_cols]
        eps = 1e-12
        s = p_neg + p_neu + p_pos + eps
        p_neg, p_neu, p_pos = p_neg/s, p_neu/s, p_pos/s

        if topic_polarity == "negative_is_pro":
            df["lean_scalar"] = (+1.0)*p_neg + 0.0*p_neu + (-1.0)*p_pos
        else:
            df["lean_scalar"] = (-1.0)*p_neg + 0.0*p_neu + (+1.0)*p_pos

        # confidence = 1 - entropy_norm
        P = np.vstack([p_neg, p_neu, p_pos]).T
        ent = -np.sum(np.where(P>0, P*np.log(P+1e-12), 0.0), axis=1)
        ent_max = math.log(3.0)
        df["conf"] = 1.0 - (ent/ent_max)
        df["_p_neg"] = p_neg
        df["_p_neu"] = p_neu
        df["_p_pos"] = p_pos
    elif label_col is not None and label_col in df.columns:
        df["lean_scalar"] = _labels_to_scalar(df[label_col].values)
        df["conf"] = 1.0
        df["_p_neg"] = 0.0
        df["_p_neu"] = 0.0
        df["_p_pos"] = 0.0
    else:
        # If label_cols provided (e.g., multiple annotators), average them
        if label_cols is None:
            raise ValueError("Provide either proba_cols or label_col (or label_cols)")
        arr = np.column_stack([_labels_to_scalar(df[c].values) for c in label_cols])
        df["lean_scalar"] = np.nanmean(arr, axis=1)
        df["conf"] = np.mean(~np.isnan(arr), axis=1)
        df["_p_neg"] = 0.0
        df["_p_neu"] = 0.0
        df["_p_pos"] = 0.0

    # Aggregate to per-user
    agg = df.groupby(user_col).agg(
        lean_scalar=("lean_scalar", "mean"),
        conf=("conf", "mean"),
        p_neg_mean=("_p_neg", "mean"),
        p_neu_mean=("_p_neu", "mean"),
        p_pos_mean=("_p_pos", "mean"),
    ).reset_index()
    agg.rename(columns={user_col: "user"}, inplace=True)
    # clamp
    agg["lean_scalar"] = agg["lean_scalar"].clip(-1.0, 1.0)
    agg["conf"] = agg["conf"].clip(0.0, 1.0)
    return agg


# --- Calibration (optional) ---

# def calibrate_probabilities(
#     df_users: pd.DataFrame,
#     method: str = "isotonic",
#     proba_cols: Tuple[str, str, str] = ("p_neg","p_neu","p_pos"),
#     y_true_col: Optional[str] = None,
#     random_state: int = 42,
# ) -> pd.DataFrame:
#     """Post-hoc recalibration of class probabilities (3-class): isotonic or temperature.

#     If your df already has calibrated probs, you can skip this.

#     - method='isotonic': apply classwise one-vs-rest isotonic regressors.
#     - method='temperature': fit a single temperature on logits estimated from probs.

#     y_true_col (optional) should contain gold labels in {-1,0,+1} if available;
#     otherwise the function performs identity (returns input).
#     """
#     if not _HAS_SKLEARN:
#         return df_users

#     req = list(proba_cols)
#     if not all(c in df_users.columns for c in req):
#         return df_users

#     if y_true_col is None or y_true_col not in df_users.columns:
#         return df_users

#     rng = np.random.RandomState(random_state)

#     P = df_users[list(proba_cols)].values.astype(float)
#     y = _labels_to_scalar(df_users[y_true_col].values)
#     y3 = np.where(y < -0.5, 0, np.where(y > 0.5, 2, 1))  # map to {0,1,2}

#     if method == "isotonic":
#         P_cal = np.zeros_like(P)
#         for k in range(3):
#             iso = IsotonicRegression(out_of_bounds="clip")
#             iso.fit(P[:,k], (y3==k).astype(float))
#             P_cal[:,k] = iso.transform(P[:,k])
#         # renormalize
#         s = P_cal.sum(axis=1, keepdims=True) + 1e-12
#         P_cal = P_cal / s
#     elif method == "temperature":
#         # Estimate logits from probs (avoid inf)
#         eps = 1e-6
#         logits = np.log(P + eps)
#         # Simple 1-param temperature scaling by minimizing Brier (approx)
#         T = 1.0
#         for _ in range(200):
#             # gradient descent on T
#             z = logits / T
#             e = np.exp(z)
#             Q = e / (e.sum(axis=1, keepdims=True) + 1e-12)
#             Y = np.eye(3)[y3]
#             grad = 2*np.mean(((Q - Y) * (-logits/(T*T))), axis=0).sum()
#             T_new = T - 0.1*grad
#             if abs(T_new - T) < 1e-6:
#                 break
#             T = max(0.05, min(20.0, T_new))
#         z = logits / T
#         e = np.exp(z)
#         P_cal = e / (e.sum(axis=1, keepdims=True) + 1e-12)
#     else:
#         return df_users

#     out = df_users.copy()
#     out[proba_cols[0]] = P_cal[:,0]
#     out[proba_cols[1]] = P_cal[:,1]
#     out[proba_cols[2]] = P_cal[:,2]
#     # recompute lean_scalar
#     out["lean_scalar"] = (-1.0)*out[proba_cols[0]] + (+1.0)*out[proba_cols[2]]
#     return out
def calibrate_probabilities(
    df_users: pd.DataFrame,
    method: str = "isotonic",
    proba_cols: Tuple[str, str, str] = ("p_neg", "p_neu", "p_pos"),
    y_true_col: Optional[str] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Post-hoc recalibration of class probabilities (3-class): isotonic or temperature.

    If your df already has calibrated probs, you can skip this.

    - method='isotonic': apply classwise one-vs-rest isotonic regressors.
    - method='temperature': fit a single temperature on logits estimated from probs.

    y_true_col (optional) should contain gold labels in {-1,0,+1} if available;
    otherwise the function performs identity (returns input).
    """
    if not _HAS_SKLEARN:
        return df_users

    req = list(proba_cols)
    if not all(c in df_users.columns for c in req):
        return df_users

    if y_true_col is None or y_true_col not in df_users.columns:
        return df_users

    rng = np.random.RandomState(random_state)

    # Filter out rows with missing values
    mask = df_users[list(proba_cols) + [y_true_col]].notna().all(axis=1)
    if mask.sum() == 0:
        return df_users
    
    df_valid = df_users[mask].copy()
    
    P = df_valid[list(proba_cols)].values.astype(float)
    y = _labels_to_scalar(df_valid[y_true_col].values)
    y3 = np.where(y < -0.5, 0, np.where(y > 0.5, 2, 1))  # map to {0,1,2}

    # Ensure probabilities are valid (sum to 1, non-negative)
    P = np.clip(P, 1e-12, 1.0)
    P = P / P.sum(axis=1, keepdims=True)

    if method == "isotonic":
        P_cal = np.zeros_like(P)
        for k in range(3):
            # Check if we have both classes for calibration
            if np.sum(y3 == k) > 0 and np.sum(y3 != k) > 0:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(P[:, k], (y3 == k).astype(float))
                P_cal[:, k] = iso.transform(P[:, k])
            else:
                # Not enough data for calibration, keep original
                P_cal[:, k] = P[:, k]
        
        # Renormalize
        P_cal = np.clip(P_cal, 1e-12, 1.0)
        s = P_cal.sum(axis=1, keepdims=True)
        P_cal = P_cal / s
        
    elif method == "temperature":
        # Estimate logits from probs (avoid inf)
        eps = 1e-8
        P_clipped = np.clip(P, eps, 1 - eps)
        logits = np.log(P_clipped)
        
        # Simple 1-param temperature scaling by minimizing cross-entropy
        T = 1.0
        best_loss = float('inf')
        patience = 0
        
        for iteration in range(200):
            # Compute calibrated probabilities
            z = logits / max(T, 0.05)  # Prevent division by very small T
            z = z - z.max(axis=1, keepdims=True)  # Numerical stability
            e = np.exp(z)
            Q = e / e.sum(axis=1, keepdims=True)
            
            # One-hot encode true labels
            Y = np.eye(3)[y3]
            
            # Cross-entropy loss
            loss = -np.mean(Y * np.log(Q + 1e-12))
            
            # Gradient of cross-entropy w.r.t. temperature
            grad = np.mean((Q - Y) * (-logits / (T * T)))
            
            # Update temperature with learning rate
            lr = 0.01
            T_new = T - lr * grad
            T_new = np.clip(T_new, 0.1, 10.0)  # Constrain T to reasonable range
            
            # Check for convergence
            if abs(T_new - T) < 1e-6:
                break
                
            # Early stopping if loss increases
            if loss < best_loss:
                best_loss = loss
                patience = 0
            else:
                patience += 1
                if patience > 10:
                    break
            
            T = T_new
        
        # Final calibrated probabilities
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        P_cal = e / e.sum(axis=1, keepdims=True)
        
    else:
        return df_users

    # Update dataframe with calibrated probabilities
    out = df_users.copy()
    out.loc[mask, proba_cols[0]] = P_cal[:, 0]
    out.loc[mask, proba_cols[1]] = P_cal[:, 1]
    out.loc[mask, proba_cols[2]] = P_cal[:, 2]
    
    # Recompute lean_scalar only for calibrated rows
    out.loc[mask, "lean_scalar"] = (-1.0) * P_cal[:, 0] + (+1.0) * P_cal[:, 2]
    
    return out

# --- Step 1: Calibration with CV + Temperature Scaling + ECE ---

def calibrate_with_cv(
    df: pd.DataFrame,
    proba_cols: Tuple[str, str, str],
    label_col: str,
    method: str = "isotonic",
    cv_folds: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Calibrate probabilities using stratified K-Fold cross-validation.

    Unlike fit-on-all-data, this avoids circular calibration by fitting
    on training folds and applying on validation folds.

    Parameters
    ----------
    df : DataFrame with probability columns and label column
    proba_cols : tuple of 3 column names for (neg, neu, pos) probabilities
    label_col : column with true sentiment labels
    method : 'isotonic' or 'sigmoid' (Platt scaling)
    cv_folds : number of stratified folds
    random_state : random seed

    Returns
    -------
    (calibrated_df, metrics_dict)
        calibrated_df has updated probability columns
        metrics_dict contains ECE, log_loss, brier_score
    """
    if not _HAS_SKLEARN:
        raise ImportError("scikit-learn required for calibration")

    result = df.copy()
    sent_map = {
        'negative': 0, 'neg': 0, 'contra': 0,
        'neutral': 1, 'neu': 1,
        'positive': 2, 'pos': 2, 'pro': 2,
    }
    y_class = result[label_col].astype(str).str.lower().map(
        lambda x: sent_map.get(x, 1)
    ).values

    P = result[list(proba_cols)].values.astype(float)
    P = np.clip(P, 1e-12, 1.0)
    P = P / P.sum(axis=1, keepdims=True)

    P_cal = np.zeros_like(P)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    for train_idx, val_idx in skf.split(P, y_class):
        P_train, y_train = P[train_idx], y_class[train_idx]
        P_val = P[val_idx]

        for k in range(3):
            y_bin_train = (y_train == k).astype(float)

            if np.sum(y_bin_train) == 0 or np.sum(y_bin_train) == len(y_bin_train):
                P_cal[val_idx, k] = P_val[:, k]
                continue

            if method == "isotonic":
                cal = IsotonicRegression(out_of_bounds="clip")
                cal.fit(P_train[:, k], y_bin_train)
                P_cal[val_idx, k] = cal.transform(P_val[:, k])
            elif method == "sigmoid":
                # Platt scaling: logistic regression on single feature
                lr = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
                lr.fit(P_train[:, k].reshape(-1, 1), y_bin_train)
                P_cal[val_idx, k] = lr.predict_proba(P_val[:, k].reshape(-1, 1))[:, 1]
            else:
                P_cal[val_idx, k] = P_val[:, k]

    # Renormalize
    P_cal = np.clip(P_cal, 1e-12, 1.0)
    P_cal = P_cal / P_cal.sum(axis=1, keepdims=True)

    # Compute metrics
    Y_onehot = np.eye(3)[y_class]
    ece, _, _, _ = compute_ece(y_class, P_cal)
    log_loss_val = -np.mean(np.sum(Y_onehot * np.log(P_cal + 1e-12), axis=1))
    brier = np.mean(np.sum((Y_onehot - P_cal) ** 2, axis=1))

    metrics = {"ece": ece, "log_loss": log_loss_val, "brier_score": brier, "method": method}

    # Update result
    for i, col in enumerate(proba_cols):
        result[col] = P_cal[:, i]

    return result, metrics


def temperature_scaling(
    y_true: np.ndarray,
    logits: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Optimize temperature T via scipy minimize_scalar minimizing NLL.

    Parameters
    ----------
    y_true : array of class indices (0, 1, 2)
    logits : array of shape (n, 3) with log-probabilities or raw logits

    Returns
    -------
    (optimal_T, calibrated_probs)
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy required for temperature scaling")

    Y_onehot = np.eye(3)[y_true.astype(int)]

    def nll(T):
        z = logits / max(T, 1e-4)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        Q = e / e.sum(axis=1, keepdims=True)
        return -np.mean(np.sum(Y_onehot * np.log(Q + 1e-12), axis=1))

    result = minimize_scalar(nll, bounds=(0.05, 10.0), method='bounded')
    T_opt = result.x

    z = logits / T_opt
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    P_cal = e / e.sum(axis=1, keepdims=True)

    return float(T_opt), P_cal


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Expected Calibration Error.

    Parameters
    ----------
    y_true : array of class indices (0, 1, 2) or binary labels
    y_prob : array of predicted probabilities (n,) for binary or (n, K) for multi-class

    Returns
    -------
    (ece, bin_accuracies, bin_confidences, bin_counts)
    """
    if y_prob.ndim == 2:
        # Multi-class: use max predicted probability and predicted class
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true).astype(float)
    else:
        confidences = y_prob
        accuracies = y_true.astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = mask.sum()
        bin_counts[i] = count
        if count > 0:
            bin_accs[i] = accuracies[mask].mean()
            bin_confs[i] = confidences[mask].mean()

    total = len(y_true)
    ece = np.sum(np.abs(bin_accs - bin_confs) * bin_counts) / max(total, 1)

    return float(ece), bin_accs, bin_confs, bin_counts


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob_before: np.ndarray,
    y_prob_after: np.ndarray,
    n_bins: int = 15,
    title_before: str = "Before Calibration",
    title_after: str = "After Calibration",
):
    """Plot reliability diagrams before and after calibration side by side."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, y_prob, title in [(axes[0], y_prob_before, title_before),
                               (axes[1], y_prob_after, title_after)]:
        ece, bin_accs, bin_confs, bin_counts = compute_ece(y_true, y_prob, n_bins)
        bin_centers = (np.arange(n_bins) + 0.5) / n_bins
        mask = bin_counts > 0

        ax.bar(bin_centers[mask], bin_accs[mask], width=1.0/n_bins,
               alpha=0.6, edgecolor='black', label='Accuracy')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')
        ax.scatter(bin_confs[mask], bin_accs[mask], c='blue', s=30, zorder=5)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{title}\nECE = {ece:.4f}', fontweight='bold')
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# --- Step 2: Sentiment Comparison Methods ---

def _translate_texts(texts: List[str], src: str = 'id', dest: str = 'en', batch_size: int = 50) -> List[str]:
    """Translate texts from src to dest language using deep_translator.
    Falls back to original text on failure.
    """
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        print("WARNING: deep_translator not installed. Run: pip install deep-translator")
        print("         Returning original (untranslated) texts.")
        return texts

    translated = []
    translator = GoogleTranslator(source=src, target=dest)
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Translating ({src}→{dest})", leave=False):
        batch = texts[i:i + batch_size]
        for t in batch:
            try:
                clean = str(t).strip()[:4999]  # Google Translate char limit
                if not clean:
                    translated.append(clean)
                    continue
                result = translator.translate(clean)
                translated.append(result if result else clean)
            except Exception:
                translated.append(str(t))
    return translated


def sentiment_vader(texts: pd.Series, translate: bool = True, src_lang: str = 'id') -> pd.DataFrame:
    """Classify sentiment using VADER.

    Parameters
    ----------
    translate : bool
        If True (default), translate texts to English before analysis.
        Essential for non-English text since VADER uses an English lexicon.
    src_lang : str
        Source language code for translation (default: 'id' for Indonesian).

    Returns DataFrame with columns: vader_label, vader_compound, vader_neg, vader_neu, vader_pos
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    text_list = [str(t) for t in texts]
    if translate:
        text_list = _translate_texts(text_list, src=src_lang, dest='en')

    results = []
    for text in tqdm(text_list, desc="VADER sentiment", leave=False):
        scores = analyzer.polarity_scores(str(text))
        compound = scores['compound']
        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        results.append({
            'vader_label': label,
            'vader_compound': compound,
            'vader_neg': scores['neg'],
            'vader_neu': scores['neu'],
            'vader_pos': scores['pos'],
        })
    return pd.DataFrame(results)


def sentiment_sentiwordnet(texts: pd.Series, translate: bool = True, src_lang: str = 'id') -> pd.DataFrame:
    """Classify sentiment using SentiWordNet.

    Parameters
    ----------
    translate : bool
        If True (default), translate texts to English before analysis.
        Essential for non-English text since SentiWordNet uses English WordNet.
    src_lang : str
        Source language code for translation (default: 'id' for Indonesian).

    Returns DataFrame with columns: swn_label, swn_neg, swn_neu, swn_pos
    """
    import nltk
    from nltk.corpus import sentiwordnet as swn
    from nltk.corpus import wordnet

    # POS tag mapping for WordNet
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        return None

    text_list = [str(t) for t in texts]
    if translate:
        text_list = _translate_texts(text_list, src=src_lang, dest='en')

    results = []
    for text in tqdm(text_list, desc="SentiWordNet sentiment", leave=False):
        try:
            tokens = nltk.word_tokenize(str(text).lower())
            tagged = nltk.pos_tag(tokens)

            pos_score = 0.0
            neg_score = 0.0
            count = 0

            for word, tag in tagged:
                wn_tag = get_wordnet_pos(tag)
                if wn_tag is None:
                    continue
                synsets = wordnet.synsets(word, pos=wn_tag)
                if not synsets:
                    continue
                syn = synsets[0]
                swn_syn = swn.senti_synset(syn.name())
                pos_score += swn_syn.pos_score()
                neg_score += swn_syn.neg_score()
                count += 1

            if count > 0:
                pos_avg = pos_score / count
                neg_avg = neg_score / count
                neu_avg = 1.0 - pos_avg - neg_avg
            else:
                pos_avg = neg_avg = 0.0
                neu_avg = 1.0

            total = pos_avg + neg_avg + neu_avg
            if total > 0:
                pos_avg /= total
                neg_avg /= total
                neu_avg /= total

            if pos_avg > neg_avg and pos_avg > neu_avg:
                label = 'positive'
            elif neg_avg > pos_avg and neg_avg > neu_avg:
                label = 'negative'
            else:
                label = 'neutral'

        except Exception:
            label, neg_avg, neu_avg, pos_avg = 'neutral', 0.0, 1.0, 0.0

        results.append({
            'swn_label': label,
            'swn_neg': neg_avg,
            'swn_neu': neu_avg,
            'swn_pos': pos_avg,
        })
    return pd.DataFrame(results)


def sentiment_llm(
    texts: pd.Series,
    api_key: str,
    model: str = "gemini-2.0-flash",
    batch_size: int = 20,
) -> pd.DataFrame:
    """Classify sentiment using Google Generative AI API.

    Returns DataFrame with column: llm_label
    """
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(model)

    labels = []
    text_list = list(texts)

    for i in tqdm(range(0, len(text_list), batch_size), desc="LLM sentiment", leave=False):
        batch = text_list[i:i + batch_size]
        prompt_lines = []
        for j, t in enumerate(batch):
            prompt_lines.append(f"{j+1}. {str(t)[:500]}")

        prompt = (
            "Classify each text below as exactly one of: negative, neutral, positive.\n"
            "Return ONLY the labels, one per line, in the same order.\n\n"
            + "\n".join(prompt_lines)
        )

        try:
            response = gen_model.generate_content(prompt)
            response_lines = response.text.strip().split('\n')
            for j, line in enumerate(response_lines[:len(batch)]):
                line_clean = line.strip().lower()
                if 'negative' in line_clean:
                    labels.append('negative')
                elif 'positive' in line_clean:
                    labels.append('positive')
                else:
                    labels.append('neutral')
            # Pad if response was shorter
            while len(labels) < i + len(batch):
                labels.append('neutral')
        except Exception:
            labels.extend(['neutral'] * len(batch))

    return pd.DataFrame({'llm_label': labels[:len(text_list)]})


def sentiment_custom_lexicon(
    texts: pd.Series,
    pos_words: List[str],
    neg_words: List[str],
) -> pd.DataFrame:
    """Simple lexicon-based sentiment classification.

    Score = (n_pos - n_neg) / (n_pos + n_neg + 1)

    Returns DataFrame with columns: lex_label, lex_score
    """
    pos_set = set(w.lower() for w in pos_words)
    neg_set = set(w.lower() for w in neg_words)

    results = []
    for text in texts:
        words = str(text).lower().split()
        n_pos = sum(1 for w in words if w in pos_set)
        n_neg = sum(1 for w in words if w in neg_set)
        score = (n_pos - n_neg) / (n_pos + n_neg + 1)

        if score > 0.05:
            label = 'positive'
        elif score < -0.05:
            label = 'negative'
        else:
            label = 'neutral'

        results.append({'lex_label': label, 'lex_score': score})
    return pd.DataFrame(results)


def sentiment_indobert(
    texts: pd.Series,
    model_name: str = "mdhugol/indonesia-bert-sentiment-classification",
    batch_size: int = 32,
    max_length: int = 128,
) -> pd.DataFrame:
    """Classify sentiment using IndoBERT (pretrained Indonesian BERT).

    Uses a HuggingFace model fine-tuned for Indonesian 3-class sentiment.
    No translation needed — works directly on Indonesian text.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID. Default: mdhugol/indonesia-bert-sentiment-classification
    batch_size : int
        Inference batch size (default 32).
    max_length : int
        Max token length per text (default 128).

    Returns DataFrame with columns: indobert_label, indobert_neg, indobert_neu, indobert_pos
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except ImportError:
        raise ImportError(
            "transformers and torch required. Run: pip install transformers torch"
        )

    import logging as _logging
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Suppress "position_ids UNEXPECTED" warning from older checkpoints
    _prev_level = _logging.getLogger("transformers.modeling_utils").level
    _logging.getLogger("transformers.modeling_utils").setLevel(_logging.ERROR)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    _logging.getLogger("transformers.modeling_utils").setLevel(_prev_level)
    model.eval()

    # Label mapping — ensure integer keys & normalize to standard labels
    _LABEL_NORM = {
        'positive': 'positive', 'pos': 'positive', 'pro': 'positive',
        'label_0': 'positive',
        'negative': 'negative', 'neg': 'negative', 'contra': 'negative',
        'label_2': 'negative',
        'neutral': 'neutral', 'neu': 'neutral',
        'label_1': 'neutral',
    }
    raw_id2label = model.config.id2label if hasattr(model.config, 'id2label') else {0: 'positive', 1: 'neutral', 2: 'negative'}
    id2label = {}
    for k, v in raw_id2label.items():
        v_low = str(v).lower().strip()
        id2label[int(k)] = _LABEL_NORM.get(v_low, v_low)

    text_list = [str(t)[:512] for t in texts]
    all_probs = []
    all_labels = []

    for i in tqdm(range(0, len(text_list), batch_size), desc="IndoBERT sentiment", leave=False):
        batch = text_list[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
        for j in range(len(batch)):
            label = id2label.get(int(preds[j]), 'neutral')
            all_labels.append(label)
            all_probs.append(probs[j])

    # Build probability columns mapped to neg/neu/pos
    results = []
    for label, probs in zip(all_labels, all_probs):
        prob_map = {}
        for idx, prob in enumerate(probs):
            prob_map[id2label.get(idx, 'neutral')] = float(prob)
        results.append({
            'indobert_label': label,
            'indobert_neg': prob_map.get('negative', 0.0),
            'indobert_neu': prob_map.get('neutral', 0.0),
            'indobert_pos': prob_map.get('positive', 0.0),
        })

    return pd.DataFrame(results)


def evaluate_sentiment_methods(
    reference: pd.Series,
    predictions_dict: Dict[str, pd.Series],
) -> pd.DataFrame:
    """Evaluate multiple sentiment methods against a reference.

    Parameters
    ----------
    reference : Series of true labels (negative/neutral/positive)
    predictions_dict : dict of {method_name: Series of predicted labels}

    Returns
    -------
    DataFrame with columns: method, accuracy, macro_f1, f1_neg, f1_neu, f1_pos,
                            precision_macro, recall_macro
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

    ref = reference.astype(str).str.lower()
    rows = []

    for method_name, preds in predictions_dict.items():
        pred = preds.astype(str).str.lower()
        # Align labels
        labels = ['negative', 'neutral', 'positive']

        acc = accuracy_score(ref, pred)
        f1_macro = f1_score(ref, pred, labels=labels, average='macro', zero_division=0)
        f1_per = f1_score(ref, pred, labels=labels, average=None, zero_division=0)
        prec = precision_score(ref, pred, labels=labels, average='macro', zero_division=0)
        rec = recall_score(ref, pred, labels=labels, average='macro', zero_division=0)
        cm = confusion_matrix(ref, pred, labels=labels)

        rows.append({
            'method': method_name,
            'accuracy': acc,
            'macro_f1': f1_macro,
            'f1_neg': f1_per[0],
            'f1_neu': f1_per[1],
            'f1_pos': f1_per[2],
            'precision_macro': prec,
            'recall_macro': rec,
            'confusion_matrix': cm,
        })

    return pd.DataFrame(rows)


# --- Step 3: User Grouping + Enhanced Leaning ---

def aggregate_user_stats(
    content: pd.DataFrame,
    user_col: str = "user",
    proba_cols: Tuple[str, str, str] = ("p_neg", "p_neu", "p_pos"),
) -> pd.DataFrame:
    """Compute diagnostic per-user statistics from content probabilities.

    Returns DataFrame with columns:
        user, n_posts, lean_scalar, lean_std, conf, p_neg_mean, p_neu_mean,
        p_pos_mean, dominant_sentiment
    """
    df = content.copy()
    p_neg, p_neu, p_pos = [df[c].astype(float).values for c in proba_cols]
    eps = 1e-12
    s = p_neg + p_neu + p_pos + eps
    p_neg, p_neu, p_pos = p_neg / s, p_neu / s, p_pos / s
    df["lean_scalar"] = (-1.0) * p_neg + 0.0 * p_neu + (+1.0) * p_pos

    P = np.vstack([p_neg, p_neu, p_pos]).T
    ent = -np.sum(np.where(P > 0, P * np.log(P + 1e-12), 0.0), axis=1)
    ent_max = math.log(3.0)
    df["conf"] = 1.0 - (ent / ent_max)

    df["_p_neg"] = p_neg
    df["_p_neu"] = p_neu
    df["_p_pos"] = p_pos

    # Dominant sentiment per post
    labels = np.array(['negative', 'neutral', 'positive'])
    df["_dominant"] = labels[np.argmax(P, axis=1)]

    agg = df.groupby(user_col).agg(
        n_posts=(user_col, "count"),
        lean_scalar=("lean_scalar", "mean"),
        lean_std=("lean_scalar", "std"),
        conf=("conf", "mean"),
        p_neg_mean=("_p_neg", "mean"),
        p_neu_mean=("_p_neu", "mean"),
        p_pos_mean=("_p_pos", "mean"),
        dominant_sentiment=("_dominant", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "neutral"),
    ).reset_index()
    agg.rename(columns={user_col: "user"}, inplace=True)
    agg["lean_scalar"] = agg["lean_scalar"].clip(-1.0, 1.0)
    agg["lean_std"] = agg["lean_std"].fillna(0.0)
    return agg


# --- Step 4: Detailed Community Classification ---

def classify_communities_detailed(
    G: nx.Graph | nx.DiGraph,
    df_users: pd.DataFrame,
    communities: Dict[str, int],
    threshold: float = 0.2,
    lean_col: str = "lean_scalar",
) -> pd.DataFrame:
    """Classify communities with detailed per-community metrics.

    Returns DataFrame with columns:
        community_id, size, stance, avg_leaning, std_leaning, median_leaning,
        pct_pro, pct_contra, pct_neutral,
        internal_edges, external_edges, modularity_contribution, confidence
    """
    lean_dict = dict(zip(df_users['user'], df_users[lean_col]))
    eps = 1e-12

    # Group users by community
    comm_members = defaultdict(list)
    for user, comm_id in communities.items():
        comm_members[comm_id].append(user)

    results = []
    for comm_id, members in comm_members.items():
        leanings = [lean_dict.get(u, np.nan) for u in members]
        leanings = [x for x in leanings if not np.isnan(x)]
        if not leanings:
            continue

        avg_lean = float(np.mean(leanings))
        std_lean = float(np.std(leanings))
        median_lean = float(np.median(leanings))

        n_pro = sum(1 for x in leanings if x > threshold)
        n_contra = sum(1 for x in leanings if x < -threshold)
        n_neutral = len(leanings) - n_pro - n_contra
        total = len(leanings)

        if avg_lean > threshold:
            stance = 'pro'
        elif avg_lean < -threshold:
            stance = 'contra'
        else:
            stance = 'neutral'

        # Edge counts
        member_set = set(members)
        internal_edges = 0
        external_edges = 0
        for u in members:
            if u not in G:
                continue
            nbrs = list(G.successors(u)) if isinstance(G, nx.DiGraph) else list(G.neighbors(u))
            for v in nbrs:
                if v in member_set:
                    internal_edges += 1
                else:
                    external_edges += 1

        mod_contrib = internal_edges / (internal_edges + external_edges + eps)
        confidence = abs(avg_lean) / (std_lean + eps)

        results.append({
            'community_id': comm_id,
            'size': len(members),
            'stance': stance,
            'avg_leaning': avg_lean,
            'std_leaning': std_lean,
            'median_leaning': median_lean,
            'pct_pro': n_pro / total,
            'pct_contra': n_contra / total,
            'pct_neutral': n_neutral / total,
            'internal_edges': internal_edges,
            'external_edges': external_edges,
            'modularity_contribution': mod_contrib,
            'confidence': confidence,
        })

    return pd.DataFrame(results).sort_values('size', ascending=False).reset_index(drop=True)


# --- Step 5: GDM Cosine Consensus ---

def cosine_similarity_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two probability vectors."""
    v1, v2 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def compute_gdm_consensus(
    G: nx.Graph | nx.DiGraph,
    df_users: pd.DataFrame,
    communities: Dict[str, int],
    proba_cols: Tuple[str, str, str] = ("p_neg_mean", "p_neu_mean", "p_pos_mean"),
) -> Dict[str, Any]:
    """Compute GDM cosine consensus as a comparison to agreement kernel ECR.

    Per edge: cosine_similarity(proba[u], proba[v])
    Classify intra vs inter based on community membership.

    Returns dict with:
        intra_cosine, inter_cosine, ecr_cosine,
        per_community_consensus (DataFrame),
        between_community_consensus (DataFrame)
    """
    # Build per-user probability vectors
    user_proba = {}
    for _, row in df_users.iterrows():
        uid = row['user']
        vals = []
        for c in proba_cols:
            v = row.get(c, np.nan)
            vals.append(v if not pd.isna(v) else 0.0)
        user_proba[uid] = np.array(vals, dtype=float)

    # Per-edge cosine similarity
    intra_sims = []
    inter_sims = []

    for u, v in G.edges():
        if u not in user_proba or v not in user_proba:
            continue
        cu = communities.get(u)
        cv = communities.get(v)
        sim = cosine_similarity_vectors(user_proba[u], user_proba[v])
        if cu is not None and cv is not None and cu == cv:
            intra_sims.append(sim)
        else:
            inter_sims.append(sim)

    intra_cosine = float(np.mean(intra_sims)) if intra_sims else float('nan')
    inter_cosine = float(np.mean(inter_sims)) if inter_sims else float('nan')
    ecr_cosine = float(intra_cosine / inter_cosine) if (
        not np.isnan(inter_cosine) and inter_cosine > 0
    ) else float('nan')

    # Per-community within-consensus
    comm_members = defaultdict(list)
    for user, cid in communities.items():
        if user in user_proba:
            comm_members[cid].append(user)

    per_comm_rows = []
    for cid, members in comm_members.items():
        if len(members) < 2:
            continue
        sims = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                u, v = members[i], members[j]
                if G.has_edge(u, v) or G.has_edge(v, u):
                    sims.append(cosine_similarity_vectors(user_proba[u], user_proba[v]))
        if sims:
            per_comm_rows.append({
                'community_id': cid,
                'size': len(members),
                'within_cosine': float(np.mean(sims)),
                'n_pairs': len(sims),
            })

    # Between-community pair consensus
    comm_ids = [c for c in comm_members if len(comm_members[c]) >= 2]
    between_rows = []
    for ca, cb in combinations(comm_ids, 2):
        sims = []
        for u in comm_members[ca]:
            for v in comm_members[cb]:
                if G.has_edge(u, v) or G.has_edge(v, u):
                    sims.append(cosine_similarity_vectors(user_proba[u], user_proba[v]))
        if sims:
            between_rows.append({
                'community_a': ca,
                'community_b': cb,
                'between_cosine': float(np.mean(sims)),
                'n_pairs': len(sims),
            })

    return {
        'intra_cosine': intra_cosine,
        'inter_cosine': inter_cosine,
        'ecr_cosine': ecr_cosine,
        'per_community_consensus': pd.DataFrame(per_comm_rows),
        'between_community_consensus': pd.DataFrame(between_rows),
    }


# --- Step 6: Diffusion Bias Quantitative Metrics ---

def _safe_linregress(x, y):
    """linregress that handles constant x (all identical values)."""
    if np.std(x) < 1e-12:
        return 0.0, float(np.mean(y)), 0.0, 1.0, 0.0
    return scipy_stats.linregress(x, y)


def compute_diffusion_bias_metrics(
    ic_df: pd.DataFrame,
    communities: Dict[str, int],
    df_users: pd.DataFrame,
    lean_col: str = "lean_scalar",
) -> Dict[str, Any]:
    """Compute quantitative diffusion bias metrics (global + per-community).

    Parameters
    ----------
    ic_df : DataFrame from simulate_diffusion_bias (columns: seed, seed_lean, infl_mean, infl_size)
    communities : dict user -> community_id
    df_users : DataFrame with user leaning

    Returns
    -------
    dict with 'global_metrics' dict and 'community_metrics' DataFrame
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy required for diffusion bias metrics")

    df = ic_df.dropna(subset=['seed_lean', 'infl_mean']).copy()

    # --- Global metrics ---
    x = df['seed_lean'].values
    y = df['infl_mean'].values

    slope, intercept, r_value, p_value, std_err = _safe_linregress(x, y)
    bias_ratio = np.mean(np.abs(y)) / max(np.mean(np.abs(x)), 1e-12)

    # Paired t-test
    t_stat, t_pvalue = scipy_stats.ttest_rel(x, y)

    # Cohen's d
    diff = y - x
    cohens_d = float(np.mean(diff) / max(np.std(diff, ddof=1), 1e-12))

    global_metrics = {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_value ** 2),
        'p_value': float(p_value),
        'std_err': float(std_err),
        'bias_ratio': float(bias_ratio),
        'ttest_stat': float(t_stat),
        'ttest_pvalue': float(t_pvalue),
        'cohens_d': cohens_d,
        'n_samples': len(df),
    }

    # --- Per-community metrics ---
    df['community'] = df['seed'].map(communities)
    comm_rows = []

    for cid, cdf in df.groupby('community'):
        if len(cdf) < 5:
            continue

        cx = cdf['seed_lean'].values
        cy = cdf['infl_mean'].values

        c_slope, c_intercept, c_r, c_p, c_se = _safe_linregress(cx, cy)

        bias = cy - cx
        # One-sample t-test: is (infl_mean - seed_lean) != 0?
        if np.std(bias) > 1e-12:
            t1_stat, t1_pvalue = scipy_stats.ttest_1samp(bias, 0)
        else:
            t1_stat, t1_pvalue = 0.0, 1.0

        # Within-ratio: fraction of influence set that is in the same community
        # (This requires seed community info which we have)
        comm_rows.append({
            'community_id': cid,
            'n_seeds': len(cdf),
            'mean_seed_lean': float(np.mean(cx)),
            'mean_infl_mean': float(np.mean(cy)),
            'bias': float(np.mean(bias)),
            'slope': float(c_slope),
            'r_squared': float(c_r ** 2),
            'ttest_stat': float(t1_stat),
            'ttest_pvalue': float(t1_pvalue),
        })

    community_metrics = pd.DataFrame(comm_rows)

    return {
        'global_metrics': global_metrics,
        'community_metrics': community_metrics,
    }


# 3) Community detection
def _nx_to_igraph(G: nx.Graph | nx.DiGraph) -> ig.Graph:
    mapping = {n:i for i,n in enumerate(G.nodes())}
    edges = [(mapping[u], mapping[v]) for u,v in G.edges()]
    g = ig.Graph(edges=edges, directed=G.is_directed())
    g.vs["name"] = list(G.nodes())
    # weights
    w = [G[u][v].get("weight", 1.0) for u,v in G.edges()]
    g.es["weight"] = w
    return g


def detect_communities(
    G: nx.Graph | nx.DiGraph,
    method: str = "leiden",
    resolution: float = 1.0,
    infomap_twolevel: bool = False,
    random_state: int = 42,
) -> Dict[str, int]:
    """Return dict user -> community_id using Leiden or Infomap. Falls back to greedy.
    """
    nodes = list(G.nodes())

    if method.lower() == "leiden" and _HAS_LEIDEN:
        g = _nx_to_igraph(G)
        part = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            weights=g.es["weight"] if "weight" in g.es.attributes() else None,
            resolution_parameter=resolution,
            seed=random_state,
        )
        comms = {g.vs[i]["name"]: int(c) for i, c in enumerate(part.membership)}
        return comms

    if method.lower() == "infomap" and _HAS_INFOMAP:
        flags = "--silent --seed {}".format(random_state)
        if G.is_directed():
            flags += " --directed"
        if infomap_twolevel:
            flags += " --two-level"
        im = Infomap(flags)
        # Map string node IDs to integers (addLink requires unsigned int)
        node_list = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        for u, v, d in G.edges(data=True):
            im.addLink(node_to_idx[u], node_to_idx[v], float(d.get("weight", 1.0)))
        im.run()
        comms = {}
        for node in im.nodes:
            comms[node_list[node.node_id]] = int(node.module_id)
        return comms

    # Fallback: greedy modularity (undirected projection)
    H = G.to_undirected()
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comm_list = list(greedy_modularity_communities(H, weight="weight"))
        comms = {}
        for cid, comm in enumerate(comm_list):
            for u in comm:
                comms[u] = cid
        return comms
    except Exception:
        # last resort: each node its own community
        return {u:i for i,u in enumerate(nodes)}


# 4) Homophily

def compute_homophily(
    G: nx.DiGraph | nx.Graph,
    df_users: pd.DataFrame,
    lean_col: str = "lean_scalar",
) -> Dict[str, float]:
    """Compute user→neighbor_mean_leaning (out-neighbors if directed). Also returns r.
    """
    lean = dict(zip(df_users["user"], df_users[lean_col]))

    neigh_mean = {}
    for u in G.nodes():
        if isinstance(G, nx.DiGraph):
            nbrs = list(G.successors(u))
        else:
            nbrs = list(G.neighbors(u))
        vals = [lean.get(v, np.nan) for v in nbrs]
        vals = [v for v in vals if not np.isnan(v)]
        neigh_mean[u] = float(np.mean(vals)) if len(vals) else np.nan

    # Pearson r (ignoring NaNs)
    a, b = [], []
    for u in G.nodes():
        lu = lean.get(u, np.nan)
        mv = neigh_mean.get(u, np.nan)
        if not (np.isnan(lu) or np.isnan(mv)):
            a.append(lu); b.append(mv)
    if len(a) >= 3:
        r = float(np.corrcoef(np.asarray(a), np.asarray(b))[0,1])
    else:
        r = float("nan")
    return {"neighbor_mean": neigh_mean, "pearson_r": r}


# 5) Diffusion bias (IC)
def _ic_run(G: nx.DiGraph, seed: str, p_activate: float, rng: random.Random) -> set:
    active = {seed}
    frontier = [seed]
    while frontier:
        u = frontier.pop()
        for v in G.successors(u):
            if v in active:
                continue
            w = float(G[u][v].get("weight", 1.0))
            p = 1.0 - (1.0 - p_activate)**max(1.0, w)  # weight-aware activation
            if rng.random() < p:
                active.add(v)
                frontier.append(v)
    return active


def simulate_diffusion_bias(
    G: nx.DiGraph | nx.Graph,
    df_users: pd.DataFrame,
    p_activate: float = 0.05,
    n_runs: int = 20,
    seed: int = 42,
    lean_col: str = "lean_scalar",
    bin_count: int = 9,
) -> pd.DataFrame:
    """IC diffusion per seed, record mean leaning of influence set vs seed leaning.

    Returns DataFrame with columns: ['seed','seed_lean','infl_mean','infl_size']
    and a summary grouped into bins can be obtained with `summarize_diffusion_bias`.
    """
    if isinstance(G, nx.Graph) and not isinstance(G, nx.DiGraph):
        # Treat as symmetric directed for diffusion
        G = nx.DiGraph(G)

    lean = dict(zip(df_users["user"], df_users[lean_col]))

    rows = []
    rng = random.Random(seed)
    nodes = list(G.nodes())
    for s in tqdm(nodes, desc="IC simulate", leave=False):
        seed_lean = lean.get(s, np.nan)
        if np.isnan(seed_lean):
            continue
        union = set()
        for _ in range(n_runs):
            union |= _ic_run(G, s, p_activate=p_activate, rng=rng)
        # remove self for influence set summary (optional)
        if s in union:
            union.remove(s)
        vals = [lean.get(v, np.nan) for v in union]
        vals = [v for v in vals if not np.isnan(v)]
        infl_mean = float(np.mean(vals)) if len(vals) else np.nan
        rows.append({"seed": s, "seed_lean": seed_lean, "infl_mean": infl_mean, "infl_size": len(vals)})

    return pd.DataFrame(rows)


def summarize_diffusion_bias(df_ic: pd.DataFrame, bin_count: int = 9) -> pd.DataFrame:
    """Bin by seed_lean and compute average infl_mean per bin to inspect closure.
    """
    df = df_ic.dropna(subset=["seed_lean","infl_mean"]).copy()
    df["bin"] = pd.cut(df["seed_lean"], bins=np.linspace(-1,1,bin_count+1), include_lowest=True)
    out = df.groupby("bin").agg(seed_lean=("seed_lean","mean"), infl_mean=("infl_mean","mean"), n=("seed","count")).reset_index(drop=True)
    return out


# 6) ECR 2.0: intra vs inter agreement
@dataclass
class ECR2Result:
    intra: float
    inter: float
    ratio: float
    homophily_r: float


def _agreement(u: float, v: float) -> float:
    """Agreement kernel in [0,1]: higher when signs/magnitudes align.
    Here use 1 - |u - v|/2 since u,v∈[-1,1]."""
    if np.isnan(u) or np.isnan(v):
        return np.nan
    return 1.0 - abs(u - v)/2.0


def compute_ecr2(
    G: nx.Graph | nx.DiGraph,
    df_users: pd.DataFrame,
    communities: Dict[str,int],
    lean_col: str = "lean_scalar",
    sample_edges: Optional[int] = None,
    random_state: int = 42,
) -> ECR2Result:
    """Compute ECR 2.0 as E[agreement | intra] / E[agreement | inter].

    - agreement(u,v) = 1 - |u - v|/2 (bounded [0,1]).
    - conditioning is on whether edge (u→v) is within the same community.
    - Optional uniform edge subsampling for scalability.
    """
    rng = np.random.RandomState(random_state)
    lean = dict(zip(df_users["user"], df_users[lean_col]))

    edges = list(G.edges())
    if sample_edges is not None and sample_edges < len(edges):
        edges = [edges[i] for i in rng.choice(len(edges), size=sample_edges, replace=False)]

    intra_vals, inter_vals = [], []
    for u, v in edges:
        cu, cv = communities.get(u), communities.get(v)
        a = _agreement(lean.get(u, np.nan), lean.get(v, np.nan))
        if np.isnan(a):
            continue
        if cu is not None and cv is not None and cu == cv:
            intra_vals.append(a)
        else:
            inter_vals.append(a)

    intra = float(np.nanmean(intra_vals)) if len(intra_vals) else float("nan")
    inter = float(np.nanmean(inter_vals)) if len(inter_vals) else float("nan")
    ratio = float(intra/inter) if (inter is not None and not np.isnan(inter) and inter>0) else float("nan")

    # Homophily r for reference
    homo = compute_homophily(G, df_users)
    homophily_r = float(homo.get("pearson_r", float("nan")))

    return ECR2Result(intra=intra, inter=inter, ratio=ratio, homophily_r=homophily_r)


# 7) Small plotting utils

def plot_diffusion_summary(df_summary: pd.DataFrame):
    """Simple matplotlib scatter for seed_lean vs infl_mean."""
    import matplotlib.pyplot as plt
    x = df_summary["seed_lean"].values
    y = df_summary["infl_mean"].values
    n = df_summary["n"].values
    plt.figure()
    plt.scatter(x, y, s=np.clip(n, 10, 200))
    plt.axline((-1,-1), (1,1), linestyle='--')
    plt.xlabel("Seed leaning (bin mean)")
    plt.ylabel("Influence-set leaning (mean)")
    plt.title("Diffusion bias summary")
    plt.show()


# 8) Convenience runner

def run_ecr2_pipeline(
    interactions: pd.DataFrame,
    content: pd.DataFrame,
    proba_cols: Optional[Tuple[str,str,str]] = None,
    label_col: Optional[str] = None,
    community_method: str = "leiden",
    resolution: float = 1.0,
    p_activate: float = 0.05,
    n_runs: int = 10,
    random_state: int = 42,
) -> Dict[str, object]:
    """One-shot helper to compute ECR 2.0 and key diagnostics.

    Returns dict with keys: G, users, comms, homophily, ic_rows, ic_bins, ecr2
    """
    G, _ = build_graph(interactions)

    if proba_cols is not None:
        users = estimate_user_leaning(content, proba_cols=proba_cols)
    elif label_col is not None:
        users = estimate_user_leaning(content, label_col=label_col)
    else:
        raise ValueError("Provide either proba_cols or label_col for leaning estimation")

    comms = detect_communities(G, method=community_method, resolution=resolution, random_state=random_state)

    homo = compute_homophily(G, users)

    ic_rows = simulate_diffusion_bias(G, users, p_activate=p_activate, n_runs=n_runs, seed=random_state)
    ic_bins = summarize_diffusion_bias(ic_rows)

    ecr2 = compute_ecr2(G, users, comms)

    return {
        "G": G,
        "users": users,
        "comms": comms,
        "homophily": homo,
        "ic_rows": ic_rows,
        "ic_bins": ic_bins,
        "ecr2": ecr2,
    }


if __name__ == "__main__":
    # Quick dry-run with a tiny synthetic graph
    import pandas as pd
    interactions = pd.DataFrame({
        'src': ['A','A','B','C','D','E','F','G','H','I'],
        'dst': ['B','C','C','D','E','F','G','H','I','A'],
        'type': ['retweet']*10,
        'weight': [1]*10,
    })
    content = pd.DataFrame({
        'user': list('ABCDEFGHI'),
        'p_neg': [0.8,0.7,0.6,0.2,0.1,0.05,0.05,0.1,0.2],
        'p_neu': [0.2,0.2,0.3,0.3,0.3,0.2,0.2,0.3,0.3],
        'p_pos': [0.0,0.1,0.1,0.5,0.6,0.75,0.75,0.6,0.5],
    })
    out = run_ecr2_pipeline(interactions, content, proba_cols=("p_neg","p_neu","p_pos"), n_runs=5)
    print("ECR2:", out['ecr2'])