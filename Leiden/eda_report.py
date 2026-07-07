"""EDA lengkap untuk 6 topik (total_data_cleaned_*.csv).
Menghasilkan tabel ringkasan (CSV) + 4 gambar di folder eda/.
Jalankan: python eda_report.py
"""
import os
import sys
import warnings
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

OUT = "eda"
os.makedirs(OUT, exist_ok=True)

TOPICS = {
    "Indonesia Vaccination": "total_data_cleaned_vaksin.csv",
    "Indonesia Gelap": "total_data_cleaned_indonesia_gelap.csv",
    "Boikot Produk Israel": "total_data_cleaned_mcd.csv",
    "Korupsi Haji": "total_data_cleaned_korupsi.csv",
    "Ijazah Jokowi": "total_data_cleaned_ijazah.csv",
    "Keracunan MBG": "total_data_cleaned_mbg.csv",
}
COLS = ["from_username", "from_id", "content", "final_sentiment",
        "reply_to_user_id", "retweet_from_user_id", "mentioned", "date_created"]
SENT = ["positive", "neutral", "negative"]
C = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c", "bar": "#3498db", "bar2": "#1abc9c"}

rows, lengths, sentiments, examples, bots = [], {}, {}, {}, {}

for name, f in TOPICS.items():
    df = pd.read_csv(f, usecols=lambda c: c in COLS, low_memory=False)
    n = len(df)
    users = df["from_username"].dropna().nunique()
    content = df["content"].fillna("").astype(str)
    L = content.str.len()
    lengths[name] = L.values

    is_bot = df["from_username"].astype(str).str.contains("bot", case=False, na=False)
    bot_acc = df.loc[is_bot, "from_username"].nunique()
    bot_rows = int(is_bot.sum())
    bots[name] = sorted(df.loc[is_bot, "from_username"].dropna().unique().tolist())[:25]

    reply = int(df["reply_to_user_id"].notna().sum())
    rt = int(df["retweet_from_user_id"].notna().sum())
    men = int(df["mentioned"].notna().sum())

    dt = pd.to_datetime(df["date_created"], format="%d/%m/%Y %H.%M.%S", errors="coerce")
    if dt.notna().mean() < 0.5:
        dt = pd.to_datetime(df["date_created"], dayfirst=True, errors="coerce")

    sc = df["final_sentiment"].astype(str).str.lower().value_counts()
    sentiments[name] = {k: int(sc.get(k, 0)) for k in SENT}

    ne = content[content.str.len() > 0]
    examples[name] = {
        "short": ne.loc[ne.str.len().idxmin()] if len(ne) else "",
        "long": ne.loc[ne.str.len().idxmax()] if len(ne) else "",
    }

    rows.append(dict(
        Topik=name, Tweet=n, Pengguna=users,
        Rata2=round(float(L.mean()), 1), Min=int(L.min()), Max=int(L.max()), Median=int(L.median()),
        AkunBot=int(bot_acc), TweetBot=bot_rows, PctBot=round(100 * bot_rows / max(n, 1), 2),
        Reply=reply, Retweet=rt, Mention=men,
        Dari=str(dt.min().date()), Sampai=str(dt.max().date()),
    ))

summary = pd.DataFrame(rows)
summary.to_csv(f"{OUT}/eda_summary.csv", index=False)
pd.DataFrame(sentiments).T.to_csv(f"{OUT}/eda_sentiment.csv")
print("=== RINGKASAN EDA ===")
print(summary.to_string(index=False))
print("\n=== AKUN BOT (contoh) ===")
for k, v in bots.items():
    print(f"  {k}: {summary.loc[summary.Topik==k,'AkunBot'].values[0]} akun -> {v[:10]}")
print("\n=== CONTOH TERPENDEK/TERPANJANG ===")
for k, v in examples.items():
    print(f"  [{k}] terpendek({len(v['short'])}): {v['short'][:60]!r}")
    print(f"  [{k}] terpanjang({len(v['long'])}): {v['long'][:80]!r}")

names = list(TOPICS.keys())
short = ["Vaksinasi", "Indo Gelap", "Boikot", "Korupsi", "Ijazah", "MBG"]

# ── Gambar 1: Volume (tweet & pengguna) ──
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(names)); w = 0.4
ax.bar(x - w/2, summary["Tweet"], w, label="Jumlah Tweet", color=C["bar"])
ax.bar(x + w/2, summary["Pengguna"], w, label="Jumlah Pengguna", color=C["bar2"])
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(short)
ax.set_ylabel("Jumlah (skala log)"); ax.set_title("Volume Data per Topik", fontweight="bold")
for i, (t, u) in enumerate(zip(summary["Tweet"], summary["Pengguna"])):
    ax.text(i - w/2, t, f"{t:,}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.text(i + w/2, u, f"{u:,}", ha="center", va="bottom", fontsize=8, rotation=90)
ax.legend(); plt.tight_layout(); fig.savefig(f"{OUT}/01_volume.png", dpi=160, facecolor="white"); plt.close(fig)

# ── Gambar 2: Distribusi panjang content (boxplot) ──
fig, ax = plt.subplots(figsize=(12, 6))
data = [np.clip(lengths[n], 0, np.percentile(lengths[n], 99)) for n in names]
bp = ax.boxplot(data, vert=True, tick_labels=short, showfliers=False, patch_artist=True)
for p in bp["boxes"]:
    p.set_facecolor("#5dade2"); p.set_alpha(0.75)
ax.set_ylabel("Panjang content (karakter)")
ax.set_title("Distribusi Panjang Content per Topik (potong di persentil 99)", fontweight="bold")
plt.tight_layout(); fig.savefig(f"{OUT}/02_content_length.png", dpi=160, facecolor="white"); plt.close(fig)

# ── Gambar 3: Distribusi sentimen (% bertumpuk) ──
fig, ax = plt.subplots(figsize=(12, 6))
bottom = np.zeros(len(names))
for s in SENT:
    vals = np.array([sentiments[n][s] for n in names], dtype=float)
    tot = np.array([sum(sentiments[n].values()) for n in names], dtype=float)
    pct = 100 * vals / np.where(tot == 0, 1, tot)
    ax.bar(short, pct, bottom=bottom, label=s.capitalize(), color=C[s])
    for i, p in enumerate(pct):
        if p > 4:
            ax.text(i, bottom[i] + p/2, f"{p:.0f}%", ha="center", va="center", fontsize=9, color="white")
    bottom += pct
ax.set_ylabel("Persentase (%)"); ax.set_ylim(0, 100)
ax.set_title("Distribusi Sentimen per Topik", fontweight="bold"); ax.legend(loc="upper right")
plt.tight_layout(); fig.savefig(f"{OUT}/03_sentiment.png", dpi=160, facecolor="white"); plt.close(fig)

# ── Gambar 4: Akun bot & tipe interaksi ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].bar(short, summary["AkunBot"], color="#e67e22")
for i, v in enumerate(summary["AkunBot"]):
    axes[0].text(i, v, str(v), ha="center", va="bottom", fontsize=9)
axes[0].set_title("Jumlah Akun Bot Terdeteksi per Topik", fontweight="bold")
axes[0].set_ylabel("Jumlah akun (username mengandung 'bot')")
xi = np.arange(len(names)); w = 0.27
axes[1].bar(xi - w, summary["Reply"], w, label="Reply", color="#3498db")
axes[1].bar(xi, summary["Retweet"], w, label="Retweet", color="#9b59b6")
axes[1].bar(xi + w, summary["Mention"], w, label="Mention", color="#1abc9c")
axes[1].set_yscale("log"); axes[1].set_xticks(xi); axes[1].set_xticklabels(short)
axes[1].set_title("Jumlah Interaksi per Tipe (skala log)", fontweight="bold")
axes[1].set_ylabel("Jumlah interaksi"); axes[1].legend()
plt.tight_layout(); fig.savefig(f"{OUT}/04_bots_interactions.png", dpi=160, facecolor="white"); plt.close(fig)

print(f"\nSaved: {OUT}/eda_summary.csv, eda_sentiment.csv, 01_volume.png, "
      f"02_content_length.png, 03_sentiment.png, 04_bots_interactions.png")
