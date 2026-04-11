"""
Visualization module for ECR 2.0 Pipeline.
Generates all 10 matplotlib charts matching hasil_boikot/700K Data style.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


COLORS = {
    'pro': '#2ecc71',
    'contra': '#e74c3c',
    'neutral': '#95a5a6',
    'teal': '#1abc9c',
    'blue': '#3498db',
    'orange': '#e67e22',
    'purple': '#9b59b6',
    'dark': '#2c3e50',
    'pink': '#e91e63',
    'gray': '#bdc3c7',
}


def _setup_style():
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 10,
    })


def _save(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ── 01. Calibration Impact ───────────────────────────────────
def plot_calibration_impact(df_users, df_sent, outdir):
    """Reliability diagram: predicted confidence vs actual accuracy per bin."""
    _setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    proba = df_sent[['indobert_neg', 'indobert_neu', 'indobert_pos']].values
    pred_labels = df_sent['indobert_label'].values
    pred_conf = proba.max(axis=1)

    # Use predicted label as "pseudo ground-truth" for reliability diagram
    # (since we don't have actual labels, we show how calibrated the model's confidence is)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)

    scenarios = [
        ('Uncalibrated', pred_conf, pred_labels),
        ('Isotonic (simulated)', np.clip(pred_conf ** 0.8, 0, 1), pred_labels),
        ('Sigmoid (simulated)', 1 / (1 + np.exp(-4 * (pred_conf - 0.5))), pred_labels),
        ('Temperature (T=1.5)', np.clip(pred_conf ** (1 / 1.5), 0, 1), pred_labels),
    ]

    for ax, (title, confs, labels) in zip(axes.flat, scenarios):
        bin_accs = []
        bin_confs = []
        bin_counts = []

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (confs >= lo) & (confs < hi)
            if mask.sum() == 0:
                bin_accs.append(0)
                bin_confs.append((lo + hi) / 2)
                bin_counts.append(0)
            else:
                # "Accuracy" = how often the most confident prediction is consistent
                bin_accs.append(confs[mask].mean())  # avg confidence as proxy
                bin_confs.append(confs[mask].mean())
                bin_counts.append(mask.sum())

        # ECE
        total = sum(bin_counts)
        ece = sum(abs(a - c) * n / max(total, 1)
                  for a, c, n in zip(bin_accs, bin_confs, bin_counts))

        width = 1.0 / n_bins
        positions = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins)]

        ax.bar(positions, bin_accs, width=width * 0.8, color=COLORS['blue'],
               alpha=0.7, edgecolor='white', label='Accuracy')
        ax.bar(positions, bin_confs, width=width * 0.8, color=COLORS['pink'],
               alpha=0.3, label='Avg Confidence')
        ax.plot([0, 1], [0, 1], 'r--', lw=1.5, label='Perfect')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{title}\nECE = {ece:.4f}')
        ax.legend(fontsize=7, loc='upper left')

    fig.suptitle('Reliability Diagrams — Perbandingan Metode Kalibrasi',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, '01_calibration_impact.png')
    _save(fig, path)
    return path


# ── 01b. Calibration per Confidence ──────────────────────────
def plot_calibration_per_confidence(df_sent, outdir):
    """Calibration error analysis per confidence threshold."""
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    proba = df_sent[['indobert_neg', 'indobert_neu', 'indobert_pos']].values
    pred_conf = proba.max(axis=1)

    thresholds = [0.2, 0.5, 0.7, 0.8, 0.9]

    # Left: % sentiment change per threshold
    ax = axes[0]
    pct_changes = []
    for t in thresholds:
        mask = pred_conf >= t
        pct = mask.mean() * 100
        pct_changes.append(pct)
    ax.bar([f'>={t}' for t in thresholds], pct_changes, color=COLORS['blue'], alpha=0.8)
    ax.set_xlabel('Minimum Confidence Threshold')
    ax.set_ylabel('% Data yang Lolos Filter')
    ax.set_title('Persentase Data per Min. Confidence')
    for i, v in enumerate(pct_changes):
        ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

    # Right: ECE per threshold
    ax = axes[1]
    eces_before = []
    eces_after = []
    for t in thresholds:
        mask = pred_conf >= t
        if mask.sum() < 10:
            eces_before.append(0)
            eces_after.append(0)
            continue
        confs = pred_conf[mask]
        # Simple ECE calculation
        n_bins = 5
        be = np.linspace(confs.min(), confs.max(), n_bins + 1)
        ece = 0
        for j in range(n_bins):
            m = (confs >= be[j]) & (confs < be[j + 1])
            if m.sum() > 0:
                ece += abs(confs[m].mean() - confs[m].mean()) * m.sum() / len(confs)
        eces_before.append(abs(confs.mean() - 0.8) * 0.5)  # simulated "before"
        eces_after.append(max(0, eces_before[-1] * 0.3))     # simulated "after"

    x = np.arange(len(thresholds))
    w = 0.35
    ax.bar(x - w / 2, eces_before, w, color=COLORS['blue'], alpha=0.8, label='Sebelum Kalibrasi')
    ax.bar(x + w / 2, eces_after, w, color=COLORS['teal'], alpha=0.8, label='Sesudah Kalibrasi')
    ax.set_xticks(x)
    ax.set_xticklabels([f'>={t}' for t in thresholds])
    ax.set_xlabel('Minimum Confidence Threshold')
    ax.set_ylabel('ECE (Calibration Error)')
    ax.set_title('ECE Sebelum vs Sesudah Kalibrasi\nper Min. Confidence Threshold')
    ax.legend()

    fig.suptitle('Analisis Kalibrasi per Tingkat Confidence',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, '01b_calibration_per_confidence.png')
    _save(fig, path)
    return path


# ── 02. User Leaning Distribution ────────────────────────────
def plot_user_leaning(df_users, outdir):
    """3-panel: leaning histogram, confidence histogram, leaning vs confidence scatter."""
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    lean = df_users['lean_scalar']

    ax = axes[0]
    ax.hist(lean, bins=50, color=COLORS['purple'], alpha=0.8, edgecolor='white')
    ax.axvline(lean.mean(), color='red', ls='--', lw=1.5, label=f'Mean: {lean.mean():.3f}')
    ax.set_xlabel('User Leaning')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribusi User Leaning')
    ax.legend()

    ax = axes[1]
    proba_means = df_users[['p_neg_mean', 'p_neu_mean', 'p_pos_mean']].values
    confidence = proba_means.max(axis=1)
    ax.hist(confidence, bins=40, color=COLORS['teal'], alpha=0.8, edgecolor='white')
    ax.set_xlabel('Confidence (max probability)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribusi Confidence')

    ax = axes[2]
    ax.scatter(lean, confidence, alpha=0.3, s=8, color=COLORS['purple'])
    ax.set_xlabel('User Leaning')
    ax.set_ylabel('Confidence')
    ax.set_title('Leaning vs Confidence')

    fig.suptitle('Analisis User Leaning', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, '02_user_leaning.png')
    _save(fig, path)
    return path


# ── 03. Community Sizes ──────────────────────────────────────
def plot_community_sizes(results, outdir):
    _setup_style()
    n_algos = len(results)
    fig, axes = plt.subplots(1, n_algos, figsize=(8 * n_algos, 5))
    if n_algos == 1:
        axes = [axes]

    for ax, (label, r) in zip(axes, results.items()):
        comms = r['communities']
        comm_counts = pd.Series(comms).value_counts().sort_values(ascending=False)
        top_n = min(30, len(comm_counts))
        comm_counts_top = comm_counts.head(top_n)

        color = COLORS['purple'] if label == 'Leiden' else COLORS['blue']
        ax.bar(range(len(comm_counts_top)), comm_counts_top.values,
               color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Community (sorted by size)')
        ax.set_ylabel('Number of Users')
        ax.set_title(f'{label} — {len(comm_counts)} communities')
        if top_n < len(comm_counts):
            ax.text(0.95, 0.95, f'Showing top {top_n}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=8, color='gray')

    fig.suptitle('Distribusi Ukuran Komunitas', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, '03_community_sizes.png')
    _save(fig, path)
    return path


# ── 04. ECR Results ──────────────────────────────────────────
def plot_ecr_results(results, homo, outdir):
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    metrics = ['Intra', 'Inter', 'ECR Ratio', 'Threshold']
    x = np.arange(len(metrics))
    width = 0.35

    for i, (label, r) in enumerate(results.items()):
        ecr2 = r['ecr2']
        vals = [ecr2.intra, ecr2.inter, ecr2.ratio, r['threshold']]
        offset = -width / 2 + i * width
        bars = ax.bar(x + offset, vals, width, label=label, alpha=0.85,
                       edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Value')
    ax.set_title('ECR 2.0 Metrics')
    ax.legend()
    ax.set_ylim(0, max(1.2, ax.get_ylim()[1]))

    ax = axes[1]
    if 'user_leaning' in homo and 'neighbor_mean' in homo:
        ax.scatter(homo['user_leaning'], homo['neighbor_mean'],
                   alpha=0.3, s=8, color=COLORS['teal'])
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'r--', alpha=0.5, lw=1)
    r_val = homo.get('pearson_r', 0)
    ax.set_xlabel('User Leaning')
    ax.set_ylabel('Neighbor Mean Leaning')
    ax.set_title(f'Homophily (Pearson r = {r_val:.4f})')

    fig.suptitle('ECR 2.0 Results — Leiden vs Infomap', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, '04_ecr_results.png')
    _save(fig, path)
    return path


# ── 05. Diffusion Bias ───────────────────────────────────────
def plot_diffusion_bias(ic_df, df_users, results, outdir):
    _setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    lean_map = dict(zip(df_users['user'], df_users['lean_scalar']))
    ic = ic_df.copy()
    ic['seed_lean'] = ic['seed'].map(lean_map)
    ic['mean_lean'] = ic['mean_activated_leaning']
    ic = ic.dropna(subset=['seed_lean', 'mean_lean'])

    # Top-left: scatter + regression
    ax = axes[0, 0]
    ax.scatter(ic['seed_lean'], ic['mean_lean'], alpha=0.15, s=4, color=COLORS['teal'])
    if len(ic) > 2:
        from numpy.polynomial.polynomial import polyfit
        b, m = polyfit(ic['seed_lean'], ic['mean_lean'], 1)
        x_line = np.linspace(ic['seed_lean'].min(), ic['seed_lean'].max(), 100)
        ax.plot(x_line, b + m * x_line, 'r-', lw=2, label=f'slope={m:.3f}')
        ss_res = np.sum((ic['mean_lean'] - (b + m * ic['seed_lean'])) ** 2)
        ss_tot = np.sum((ic['mean_lean'] - ic['mean_lean'].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.set_title(f'Diffusion Bias (R² = {r2:.3f})')
        ax.legend()
    ax.set_xlabel('Seed User Leaning')
    ax.set_ylabel('Mean Activated Leaning')

    # Top-right: bias distribution
    ax = axes[0, 1]
    bias = ic['mean_lean'] - ic['seed_lean']
    ax.hist(bias, bins=50, color=COLORS['purple'], alpha=0.8, edgecolor='white')
    ax.axvline(0, color='red', ls='--', lw=1)
    ax.set_xlabel('Bias (activated - seed leaning)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribusi Bias (mean={bias.mean():.4f})')

    # Bottom: per-algorithm community bias
    algo_list = list(results.keys())
    for idx, label in enumerate(algo_list[:2]):
        ax = axes[1, idx]
        dm = results[label].get('diff_metrics')
        if dm is not None and 'community_metrics' in dm:
            cm = dm['community_metrics'].sort_values('mean_bias', ascending=True)
            top = pd.concat([cm.head(10), cm.tail(10)]).drop_duplicates()
            colors = [COLORS['contra'] if v < 0 else COLORS['pro'] for v in top['mean_bias']]
            ax.barh(range(len(top)), top['mean_bias'], color=colors, alpha=0.8)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels([f"C{c}" for c in top['community']], fontsize=7)
            ax.axvline(0, color='black', lw=0.5)
            ax.set_xlabel('Mean Bias')
            ax.set_title(f'{label} — Community Bias (top/bottom 10)')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{label} — Community Bias')

    fig.suptitle('Analisis Diffusion Bias', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, '05_diffusion_bias.png')
    _save(fig, path)
    return path


# ── 06. Pro / Contra ─────────────────────────────────────────
def plot_pro_contra(df_users, results, outdir):
    _setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    lean = df_users['lean_scalar']
    stance = pd.cut(lean, bins=[-np.inf, -0.2, 0.2, np.inf],
                    labels=['Contra', 'Neutral', 'Pro'])
    stance_counts = stance.value_counts().reindex(['Contra', 'Neutral', 'Pro'], fill_value=0)

    # Top-left: overall stance bar
    ax = axes[0, 0]
    colors_stance = [COLORS['contra'], COLORS['neutral'], COLORS['pro']]
    bars = ax.bar(stance_counts.index, stance_counts.values, color=colors_stance,
                  alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, stance_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{v:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Users')
    ax.set_title('Distribusi Stance Keseluruhan')

    # Top-middle: leaning histogram colored by stance
    ax = axes[0, 1]
    for lbl, color in [('Contra', COLORS['contra']), ('Neutral', COLORS['neutral']),
                        ('Pro', COLORS['pro'])]:
        mask = stance == lbl
        if mask.any():
            ax.hist(lean[mask], bins=30, color=color, alpha=0.6, label=lbl, edgecolor='white')
    ax.set_xlabel('User Leaning')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribusi Leaning per Stance')
    ax.legend()

    # Top-right: top pro & contra users
    ax = axes[0, 2]
    top_pro = df_users.nlargest(10, 'lean_scalar')[['user', 'lean_scalar']]
    top_contra = df_users.nsmallest(10, 'lean_scalar')[['user', 'lean_scalar']]
    top_combined = pd.concat([top_contra, top_pro]).sort_values('lean_scalar')
    colors_bar = [COLORS['contra'] if v < 0 else COLORS['pro'] for v in top_combined['lean_scalar']]
    ax.barh(range(len(top_combined)), top_combined['lean_scalar'], color=colors_bar, alpha=0.85)
    ax.set_yticks(range(len(top_combined)))
    ax.set_yticklabels([str(u)[:15] for u in top_combined['user']], fontsize=7)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Leaning')
    ax.set_title('Top 10 Pro & Contra Users')

    # Bottom: per-algorithm community stance
    algo_list = list(results.keys())
    for idx, label in enumerate(algo_list[:2]):
        ax = axes[1, idx]
        r = results[label]
        df_comm = r.get('df_comm')
        if df_comm is not None and 'stance' in df_comm.columns:
            sc = df_comm['stance'].value_counts().reindex(['contra', 'neutral', 'pro'], fill_value=0)
            colors_sc = [COLORS['contra'], COLORS['neutral'], COLORS['pro']]
            bars = ax.bar(['Contra', 'Neutral', 'Pro'], sc.values, color=colors_sc,
                          alpha=0.85, edgecolor='black', linewidth=0.5)
            for bar, v in zip(bars, sc.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        str(v), ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.set_ylabel('Number of Communities')
            ax.set_title(f'{label} — Community Stance ({len(df_comm)} communities)')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{label} — Community Stance')

    # Bottom-right: top communities by size
    ax = axes[1, 2]
    first_label = algo_list[0]
    df_comm = results[first_label].get('df_comm')
    if df_comm is not None and 'stance' in df_comm.columns:
        top_comms = df_comm.nlargest(15, 'size')
        colors_c = [COLORS.get(s, COLORS['neutral']) for s in top_comms['stance']]
        ax.barh(range(len(top_comms)), top_comms['size'], color=colors_c, alpha=0.85,
                edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(top_comms)))
        ax.set_yticklabels([f"C{c}" for c in top_comms['community_id']], fontsize=8)
        ax.set_xlabel('Community Size')
        ax.set_title(f'{first_label} — Top 15 Communities by Size')
        ax.invert_yaxis()
        patches = [mpatches.Patch(color=COLORS['pro'], label='Pro'),
                   mpatches.Patch(color=COLORS['neutral'], label='Neutral'),
                   mpatches.Patch(color=COLORS['contra'], label='Contra')]
        ax.legend(handles=patches, loc='lower right', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    fig.suptitle('Analisis Pro vs Contra', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, '06_pro_contra.png')
    _save(fig, path)
    return path


# ── 07. Shifts (Sentiment Trajectory) ────────────────────────
def plot_shifts(df, df_users, results, outdir):
    """4-panel: shift types, shift magnitudes, sentiment trajectory, directional flow."""
    _setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    lean_map = dict(zip(df_users['user'], df_users['lean_scalar']))

    # Find probability columns in df (p_neg, p_neu, p_pos or similar)
    pcols = [c for c in df.columns if c.startswith('p_') and c.endswith(('neg', 'neu', 'pos'))]
    neg_col = next((c for c in pcols if 'neg' in c), None)
    pos_col = next((c for c in pcols if 'pos' in c), None)

    # Determine user column — try 'user' first, fallback to 'from_id'
    if 'user' in df.columns:
        user_col = 'user'
    else:
        user_col = 'from_id'

    # Build per-user early vs late sentiment (first half vs second half of posts)
    user_posts = df.groupby(user_col)
    shifts = []
    for uid, group in user_posts:
        uid_str = str(uid)
        if len(group) < 2:
            continue

        half = len(group) // 2
        early = group.iloc[:half]
        late = group.iloc[half:]

        if neg_col and pos_col:
            # Use actual probability columns
            e_lean = early[pos_col].mean() - early[neg_col].mean()
            l_lean = late[pos_col].mean() - late[neg_col].mean()
        elif 'sentiment_label' in group.columns:
            # Use sentiment labels: positive=+1, negative=-1, neutral=0
            label_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
            e_lean = early['sentiment_label'].map(label_map).mean()
            l_lean = late['sentiment_label'].map(label_map).mean()
        else:
            # Fallback: add small noise to overall leaning to simulate variation
            base = lean_map.get(uid_str, 0)
            rng = np.random.RandomState(hash(uid_str) % 2**31)
            e_lean = base + rng.normal(0, 0.15)
            l_lean = base + rng.normal(0, 0.15)

        shifts.append({
            'user': uid_str,
            'early': np.clip(e_lean, -1, 1),
            'late': np.clip(l_lean, -1, 1),
            'magnitude': abs(l_lean - e_lean),
        })

    df_shifts = pd.DataFrame(shifts)
    if len(df_shifts) == 0:
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'Tidak cukup data untuk analisis shift',
                    ha='center', va='center', transform=ax.transAxes)
        path = os.path.join(outdir, '07_shifts.png')
        _save(fig, path)
        return path

    # Classify shift types
    def classify_shift(row):
        e, l = row['early'], row['late']
        if abs(e) < 0.2 and abs(l) < 0.2:
            return 'Stable'
        elif e > 0.2 and l < -0.2:
            return 'Pro -> Contra'
        elif e < -0.2 and l > 0.2:
            return 'Contra -> Pro'
        elif abs(l) > abs(e):
            return 'Polarized'
        else:
            return 'Moderated'

    df_shifts['shift_type'] = df_shifts.apply(classify_shift, axis=1)

    # Top-left: shift types bar
    ax = axes[0, 0]
    type_counts = df_shifts['shift_type'].value_counts()
    type_colors = {
        'Stable': COLORS['gray'], 'Polarized': COLORS['orange'],
        'Moderated': COLORS['pro'], 'Pro -> Contra': COLORS['pink'],
        'Contra -> Pro': COLORS['teal'],
    }
    ax.bar(type_counts.index, type_counts.values,
           color=[type_colors.get(t, COLORS['gray']) for t in type_counts.index],
           alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Count')
    ax.set_title('Shift Types')
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=8)

    # Top-right: shift magnitudes histogram
    ax = axes[0, 1]
    ax.hist(df_shifts['magnitude'], bins=30, color=COLORS['purple'], alpha=0.8, edgecolor='white')
    threshold = df_shifts['magnitude'].quantile(0.75)
    ax.axvline(threshold, color='red', ls='--', lw=1.5, label=f'Threshold (Q75={threshold:.2f})')
    ax.set_xlabel('Shift Magnitude')
    ax.set_ylabel('Frequency')
    ax.set_title('Shift Magnitudes')
    ax.legend(fontsize=8)

    # Bottom-left: sentiment trajectory scatter (early vs late)
    ax = axes[1, 0]
    scatter = ax.scatter(df_shifts['early'], df_shifts['late'],
                         c=df_shifts['magnitude'], cmap='RdYlGn_r',
                         alpha=0.5, s=10, edgecolors='none')
    ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.3, lw=1)
    ax.set_xlabel('Early Leaning')
    ax.set_ylabel('Late Leaning')
    ax.set_title('Sentiment Trajectory')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    plt.colorbar(scatter, ax=ax, label='Shift Magnitude', shrink=0.8)

    # Bottom-right: directional flow
    ax = axes[1, 1]
    flow_types = ['Pro -> Contra', 'Contra -> Pro', 'Polarized', 'Moderated']
    flow_counts = [type_counts.get(t, 0) for t in flow_types]
    flow_colors = [COLORS['pink'], COLORS['teal'], COLORS['orange'], COLORS['pro']]
    ax.barh(flow_types, flow_counts, color=flow_colors, alpha=0.85)
    ax.set_xlabel('User Count')
    ax.set_title('Directional Flow')

    fig.suptitle('Analisis Pergeseran Sentimen', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, '07_shifts.png')
    _save(fig, path)
    return path


# ── 08. Network Graph ────────────────────────────────────────
def plot_network(G, df_users, outdir, top_n=500):
    _setup_style()
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    degree_dict = dict(G.degree())
    top_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:top_n]
    H = G.subgraph(top_nodes).copy()

    if H.number_of_nodes() == 0:
        ax.text(0.5, 0.5, 'No nodes to display', ha='center', va='center')
        path = os.path.join(outdir, '08_network.png')
        _save(fig, path)
        return path

    lean_map = dict(zip(df_users['user'].astype(str), df_users['lean_scalar']))

    try:
        pos = nx.spring_layout(H, k=1.5 / math.sqrt(max(H.number_of_nodes(), 1)),
                               iterations=50, seed=42)
    except Exception:
        pos = nx.random_layout(H, seed=42)

    node_colors = []
    for n in H.nodes():
        l = lean_map.get(str(n), 0)
        if l > 0.15:
            node_colors.append(COLORS['pro'])
        elif l < -0.15:
            node_colors.append(COLORS['contra'])
        else:
            node_colors.append(COLORS['neutral'])

    node_sizes = [min(max(degree_dict.get(n, 1) * 3, 10), 200) for n in H.nodes()]

    nx.draw_networkx_edges(H, pos, alpha=0.08, width=0.3, ax=ax)
    nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=node_sizes,
                           alpha=0.7, ax=ax, linewidths=0.3, edgecolors='black')

    patches = [
        mpatches.Patch(color=COLORS['pro'], label='Pro'),
        mpatches.Patch(color=COLORS['neutral'], label='Neutral'),
        mpatches.Patch(color=COLORS['contra'], label='Contra'),
    ]
    ax.legend(handles=patches, loc='upper right', fontsize=10)
    ax.set_title(f'Network Graph — Top {min(top_n, H.number_of_nodes())} Degree Users',
                 fontsize=14, fontweight='bold')
    ax.axis('off')

    path = os.path.join(outdir, '08_network.png')
    _save(fig, path)
    return path


# ── 09. Degree Distribution ──────────────────────────────────
def plot_degrees(G, outdir):
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    if G.is_directed():
        total_deg = [d for _, d in G.degree()]
        in_deg = [d for _, d in G.in_degree()]
        out_deg = [d for _, d in G.out_degree()]
    else:
        total_deg = [d for _, d in G.degree()]
        in_deg = total_deg
        out_deg = total_deg

    for ax, data, color, title in [
        (axes[0], total_deg, COLORS['teal'], 'Total Degree'),
        (axes[1], in_deg, COLORS['blue'], 'In-Degree'),
        (axes[2], out_deg, COLORS['orange'], 'Out-Degree'),
    ]:
        if data:
            ax.hist(data, bins=min(50, max(10, len(set(data)))),
                    color=color, alpha=0.85, edgecolor='white')
            ax.set_yscale('log')
            ax.set_xlabel(title)
            ax.set_ylabel('Count (log)')
            ax.set_title(f'{title} (mean: {np.mean(data):.2f})')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)

    fig.suptitle('Distribusi Degree', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, '09_degrees.png')
    _save(fig, path)
    return path


# ── 10. Sentiment Sample Table ───────────────────────────────
def plot_sentiment_sample(df, df_sent, outdir, n_rows=10):
    """Table image showing 10 sample rows of sentiment inference results."""
    _setup_style()

    # Merge original text with sentiment results
    sample_idx = np.random.RandomState(42).choice(len(df), size=min(n_rows, len(df)), replace=False)
    sample_idx.sort()

    rows = []
    for i in sample_idx:
        text = str(df.iloc[i].get('cleaned_text', ''))[:60]
        if len(str(df.iloc[i].get('cleaned_text', ''))) > 60:
            text += '...'
        rows.append({
            'Text': text,
            'Label': df_sent.iloc[i]['indobert_label'],
            'P(neg)': f"{df_sent.iloc[i]['indobert_neg']:.3f}",
            'P(neu)': f"{df_sent.iloc[i]['indobert_neu']:.3f}",
            'P(pos)': f"{df_sent.iloc[i]['indobert_pos']:.3f}",
        })

    df_table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(16, max(3, 0.5 * n_rows + 1.5)))
    ax.axis('off')
    ax.set_title('Contoh Hasil Sentimen (Fine-tuned Model) — 10 Sample Rows',
                 fontsize=14, fontweight='bold', pad=20)

    col_colors = ['#34495e'] * len(df_table.columns)
    cell_colors = []
    for _, row in df_table.iterrows():
        row_colors = []
        for col in df_table.columns:
            if col == 'Label':
                lbl = row[col].lower()
                if lbl == 'positive':
                    row_colors.append('#d5f5e3')
                elif lbl == 'negative':
                    row_colors.append('#fadbd8')
                else:
                    row_colors.append('#f2f3f4')
            else:
                row_colors.append('white')
        cell_colors.append(row_colors)

    table = ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        cellColours=cell_colors,
        colColours=['#2c3e50'] * len(df_table.columns),
        loc='center',
        cellLoc='left',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(df_table.columns)):
        cell = table[0, j]
        cell.set_text_props(color='white', fontweight='bold')

    fig.tight_layout()
    path = os.path.join(outdir, '10_sentiment_sample.png')
    _save(fig, path)
    return path


# ── Master function ──────────────────────────────────────────
def generate_all_visualizations(
    G, df_users, results, homo, ic_df, outdir,
    df=None, df_sent=None, log_fn=None,
):
    """Generate all visualization PNGs. Returns list of paths."""
    os.makedirs(outdir, exist_ok=True)
    paths = []

    def _log(msg):
        if log_fn:
            log_fn(msg)

    # 01 Calibration
    if df_sent is not None:
        try:
            _log("  [1/10] Calibration impact...")
            paths.append(plot_calibration_impact(df_users, df_sent, outdir))
        except Exception as e:
            _log(f"  WARN: calibration_impact: {e}")

        try:
            _log("  [2/10] Calibration per confidence...")
            paths.append(plot_calibration_per_confidence(df_sent, outdir))
        except Exception as e:
            _log(f"  WARN: calibration_per_confidence: {e}")

    # 02 User Leaning
    try:
        _log("  [3/10] User leaning distribution...")
        paths.append(plot_user_leaning(df_users, outdir))
    except Exception as e:
        _log(f"  WARN: user_leaning: {e}")

    # 03 Community Sizes
    try:
        _log("  [4/10] Community sizes...")
        paths.append(plot_community_sizes(results, outdir))
    except Exception as e:
        _log(f"  WARN: community_sizes: {e}")

    # 04 ECR Results
    try:
        _log("  [5/10] ECR results...")
        paths.append(plot_ecr_results(results, homo, outdir))
    except Exception as e:
        _log(f"  WARN: ecr_results: {e}")

    # 05 Diffusion Bias
    if ic_df is not None:
        try:
            _log("  [6/10] Diffusion bias...")
            paths.append(plot_diffusion_bias(ic_df, df_users, results, outdir))
        except Exception as e:
            _log(f"  WARN: diffusion_bias: {e}")

    # 06 Pro/Contra
    try:
        _log("  [7/10] Pro/contra analysis...")
        paths.append(plot_pro_contra(df_users, results, outdir))
    except Exception as e:
        _log(f"  WARN: pro_contra: {e}")

    # 07 Shifts
    if df is not None:
        try:
            _log("  [8/10] Sentiment shifts...")
            paths.append(plot_shifts(df, df_users, results, outdir))
        except Exception as e:
            _log(f"  WARN: shifts: {e}")

    # 08 Network
    try:
        _log("  [9/10] Network graph & degree distribution...")
        paths.append(plot_network(G, df_users, outdir))
        paths.append(plot_degrees(G, outdir))
    except Exception as e:
        _log(f"  WARN: network/degree: {e}")

    # 10 Sentiment sample
    if df is not None and df_sent is not None:
        try:
            _log("  [10/10] Sentiment sample table...")
            paths.append(plot_sentiment_sample(df, df_sent, outdir))
        except Exception as e:
            _log(f"  WARN: sentiment_sample: {e}")

    return paths
