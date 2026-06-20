"""Penentuan Isu Krisis vs Non-Krisis untuk 6 topik viral.

Indeks Krisis = kombinasi multi-indikator (komposit), bukan satu metrik.
Membaca hasil pipeline (users_leaning.csv, ecr_metrics.txt) + metrik difusi
tervalidasi (Tabel IV.13). Output: tabel skor, klasifikasi bertingkat, CSV, chart.

Jalankan: python crisis_detection.py
"""
import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "pipeline_out/new"
OUTDIR = "crisis_analysis"
os.makedirs(OUTDIR, exist_ok=True)

TOPICS = {
    "vaksin":          "Indonesia Vaccination",
    "indonesia_gelap": "Indonesia Gelap",
    "boikot":          "Boikot Produk Israel",
    "korupsi":         "Korupsi Haji",
    "ijazah":          "Ijazah Jokowi",
    "mbg":             "Keracunan MBG",
}

# Cohen's d & bias ratio difusi tervalidasi (Tabel IV.13 - effect size pro vs kontra)
DIFF = {
    "vaksin":          (0.2045, 0.7588),
    "indonesia_gelap": (1.0528, 0.7924),
    "boikot":          (1.8468, 0.9675),
    "korupsi":         (-0.0311, 0.9952),
    "ijazah":          (0.3934, 0.9771),
    "mbg":             (0.7474, 0.9816),
}

# Indikator (urutan tetap)
INDICATORS = ["severity", "intensity", "diffusion", "virality", "polarization", "echo"]

# Pembobotan SETARA / equal weighting (w_k = 1/K untuk semua k).
# Dasar: OECD/JRC (2008) menyatakan bobot pada hakikatnya value judgement;
# bila tidak ada dasar teoretis/statistik untuk membedakan bobot, equal weighting
# adalah pilihan paling transparan. Robustnya diuji pada bagian sensitivitas.
W = {k: 1.0 / len(INDICATORS) for k in INDICATORS}


def read_metrics(topic):
    ul = pd.read_csv(f"{BASE}/{topic}/users_leaning.csv")
    n = len(ul)
    p_neg = float(ul["p_neg_mean"].mean())
    abs_lean = float(ul["lean_scalar"].abs().mean())

    em = open(f"{BASE}/{topic}/leiden/ecr_metrics.txt", encoding="utf-8").read()
    ecr = float(re.search(r"ECR 2.0 ratio\s*:\s*([\d.]+)", em).group(1))
    thr = float(re.search(r"Threshold\s*:\s*([\d.]+)", em).group(1))
    homo = float(re.search(r"Homophily.*?:\s*([\-\d.]+)", em).group(1))
    cohens_d, bias_ratio = DIFF[topic]
    return dict(n=n, p_neg=p_neg, abs_lean=abs_lean, ecr=ecr, thr=thr,
               homophily=homo, cohens_d=cohens_d, bias_ratio=bias_ratio,
               echo=int(ecr > thr))


# --- kumpulkan metrik mentah ---
raw = {t: read_metrics(t) for t in TOPICS}

logn = {t: math.log10(max(raw[t]["n"], 1)) for t in TOPICS}
ln_min, ln_max = min(logn.values()), max(logn.values())
homo_max = max(abs(raw[t]["homophily"]) for t in TOPICS) or 1.0

rows = []
for t in TOPICS:
    m = raw[t]
    I = {
        "severity":     min(max(m["p_neg"], 0), 1),
        "intensity":    min(max(m["abs_lean"], 0), 1),
        "diffusion":    min(abs(m["cohens_d"]) / 0.8, 1.0),          # 0.8 = "large effect"
        "virality":     (logn[t] - ln_min) / (ln_max - ln_min + 1e-9),
        "polarization": min(abs(m["homophily"]) / homo_max, 1.0),
        "echo":         float(m["echo"]),
    }
    score = sum(W[k] * I[k] for k in W)
    rows.append(dict(topic=t, name=TOPICS[t], **m, **{f"I_{k}": I[k] for k in I},
                     crisis_index=score))

df = pd.DataFrame(rows).sort_values("crisis_index", ascending=False).reset_index(drop=True)


def natural_breaks(scores, k=3, seed=0):
    """Ambang kelas via natural breaks (k-means 1D / Jenks).
    Mengembalikan (k-1) batas = titik tengah pada celah antar-kelas yang
    bersebelahan. Ambang ditentukan OLEH DATA, bukan ditetapkan manual.
    """
    from sklearn.cluster import KMeans
    s = np.sort(np.asarray(scores, dtype=float))
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(
        s.reshape(-1, 1)).labels_
    bounds = sorted((s[i - 1] + s[i]) / 2.0
                    for i in range(1, len(s)) if labels[i] != labels[i - 1])
    return bounds


# Ambang kelas dihitung OTOMATIS dari distribusi Indeks Krisis (natural breaks,
# k=3). Tiga tingkat dipilih untuk meniru skema kelas keparahan bertingkat
# (mis. INFORM Severity Index). Tidak ada ambang yang ditetapkan manual.
_breaks = natural_breaks(df["crisis_index"].to_numpy(), k=3)
THETA1, THETA2 = _breaks[0], _breaks[1]
print(f"Ambang natural breaks (k=3): theta1={THETA1:.3f}, theta2={THETA2:.3f}")


def tier(s, t1=None, t2=None):
    t1 = THETA1 if t1 is None else t1
    t2 = THETA2 if t2 is None else t2
    if s >= t2:
        return "KRISIS TINGGI"
    if s >= t1:
        return "KRISIS"
    return "NON-KRISIS"


df["kategori"] = df["crisis_index"].apply(tier)

# --- output tabel ---
pd.set_option("display.width", 160)
cols = ["name", "n", "p_neg", "abs_lean", "cohens_d", "homophily", "echo",
        "crisis_index", "kategori"]
print("\n=== INDEKS KRISIS 6 TOPIK ===")
print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

df.to_csv(f"{OUTDIR}/crisis_scores.csv", index=False)
print(f"\nSaved {OUTDIR}/crisis_scores.csv")

# --- chart ---
fig, ax = plt.subplots(figsize=(12, 7))
colors = {"KRISIS TINGGI": "#c0392b", "KRISIS": "#e67e22", "NON-KRISIS": "#27ae60"}
d2 = df.sort_values("crisis_index")
bars = ax.barh(d2["name"], d2["crisis_index"],
               color=[colors[k] for k in d2["kategori"]], edgecolor="black", alpha=0.9)
for b, s, k in zip(bars, d2["crisis_index"], d2["kategori"]):
    ax.text(s + 0.01, b.get_y() + b.get_height() / 2, f"{s:.2f}  ({k})",
            va="center", fontsize=11, fontweight="bold")
ax.axvline(THETA1, ls="--", color="#e67e22", alpha=0.7,
           label=f"θ₁={THETA1:.2f} (natural breaks)")
ax.axvline(THETA2, ls="--", color="#c0392b", alpha=0.7,
           label=f"θ₂={THETA2:.2f} (natural breaks)")
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim(0, 1.0)
ax.set_xlabel("Indeks Krisis (komposit)", fontsize=12)
ax.set_title("Penentuan Isu Krisis - 6 Topik Viral", fontsize=15, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/crisis_index.png", dpi=180, facecolor="white")
print(f"Saved {OUTDIR}/crisis_index.png")


# ============================================================
# UJI SENSITIVITAS BOBOT  (OECD/JRC 2008, Step 7: Robustness)
# Pertanyaan: apakah kategori krisis stabil meski bobot diubah-ubah?
# ============================================================
print("\n=== UJI SENSITIVITAS BOBOT ===")
I_mat = df[[f"I_{k}" for k in INDICATORS]].to_numpy()   # (topik x indikator)
names = df["name"].tolist()
base_cat = df["kategori"].tolist()

N = 5000
rng = np.random.default_rng(42)
# Skenario A: seluruh ruang bobot (Dirichlet uniform di simpleks, w>=0, sum=1)
W_uniform = rng.dirichlet(np.ones(len(INDICATORS)), size=N)
# Skenario B: perturbasi lokal di sekitar equal weighting
W_local = np.clip(1.0 / len(INDICATORS) + rng.normal(0, 0.10, (N, len(INDICATORS))), 1e-6, None)
W_local = W_local / W_local.sum(axis=1, keepdims=True)


def summarize(Wset, label):
    scores = Wset @ I_mat.T                  # (N x topik)
    cats = np.vectorize(tier)(scores)
    print(f"\n-- {label} (N={N}) --")
    out = []
    for j, nm in enumerate(names):
        col = cats[:, j]
        stab = float(np.mean(col == base_cat[j]))     # % sama dgn equal-weight
        uniq, cnt = np.unique(col, return_counts=True)
        modal = uniq[int(np.argmax(cnt))]
        smin, smean, smax = scores[:, j].min(), scores[:, j].mean(), scores[:, j].max()
        flag = "STABIL" if stab >= 0.9 else ("rapuh" if stab < 0.6 else "cukup stabil")
        print(f"  {nm:22s} base={base_cat[j]:13s} stabil={stab*100:5.1f}%  "
              f"indeks[{smin:.2f}-{smax:.2f}] -> {flag}")
        out.append(dict(name=nm, base=base_cat[j], modal=modal, stability=stab,
                        idx_min=smin, idx_mean=smean, idx_max=smax))
    return pd.DataFrame(out)


s_uniform = summarize(W_uniform, "Skenario A: seluruh ruang bobot (Dirichlet uniform)")
s_local = summarize(W_local, "Skenario B: perturbasi lokal di sekitar equal weighting")
s_uniform.to_csv(f"{OUTDIR}/crisis_sensitivity_uniform.csv", index=False)
s_local.to_csv(f"{OUTDIR}/crisis_sensitivity_local.csv", index=False)
print(f"\nSaved {OUTDIR}/crisis_sensitivity_uniform.csv & _local.csv")

# --- boxplot: rentang indeks tiap topik di seluruh ruang bobot (skenario A) ---
fig, ax = plt.subplots(figsize=(12, 7))
order = df.sort_values("crisis_index")["name"].tolist()
scores_uniform = W_uniform @ I_mat.T
data_box = [scores_uniform[:, names.index(nm)] for nm in order]
bp = ax.boxplot(data_box, vert=False, labels=order, showfliers=False, patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor("#5dade2")
    patch.set_alpha(0.7)
for i, nm in enumerate(order, start=1):
    base_idx = float(df.loc[df["name"] == nm, "crisis_index"].values[0])
    ax.plot(base_idx, i, "D", color="black", markersize=7,
            label="Equal weighting" if i == 1 else None)
ax.axvline(THETA1, ls="--", color="#e67e22", alpha=0.8)
ax.axvline(THETA2, ls="--", color="#c0392b", alpha=0.8)
ax.set_xlim(0, 1.0)
ax.set_xlabel("Indeks Krisis di bawah variasi bobot (Dirichlet uniform)", fontsize=12)
ax.set_title("Uji Sensitivitas Bobot - Stabilitas Klasifikasi Krisis",
             fontsize=15, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig(f"{OUTDIR}/crisis_sensitivity.png", dpi=180, facecolor="white")
print(f"Saved {OUTDIR}/crisis_sensitivity.png")
