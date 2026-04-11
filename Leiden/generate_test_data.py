"""Generate a small synthetic test dataset for the ECR 2.0 pipeline GUI."""

import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

N_USERS = 150
N_TWEETS = 800

# Create users with IDs
users = [{"id": str(1000 + i), "username": f"user_{i}"} for i in range(N_USERS)]

# Assign users to 3 latent groups (pro, neutral, contra) to create community structure
group_sizes = [60, 40, 50]
groups = []
for g, sz in enumerate(group_sizes):
    groups.extend([g] * sz)
random.shuffle(groups)
for i, u in enumerate(users):
    u["group"] = groups[i]

# Sample Indonesian-like tweet texts per group
pro_texts = [
    "boikot produk ini karena tidak mendukung rakyat",
    "kita harus berhenti membeli dari mereka",
    "dukung produk lokal, tinggalkan yang merugikan",
    "sudah saatnya kita beralih ke alternatif lain",
    "mereka tidak peduli dengan konsumen, boikot saja",
    "jangan beli lagi, cari yang lebih baik",
    "gerakan boikot ini harus terus dilanjutkan",
    "saatnya rakyat bersatu melawan ketidakadilan",
    "produk mereka sudah tidak layak didukung",
    "boikot adalah hak kita sebagai konsumen",
]
neutral_texts = [
    "belum tau mau beli apa hari ini",
    "lagi rame ya soal boikot, entahlah",
    "masing-masing punya hak pilih sendiri",
    "yang penting harganya terjangkau",
    "nggak ikut-ikutan, beli sesuai kebutuhan aja",
    "semua merek punya kelebihan dan kekurangan",
    "biarkan konsumen memilih sendiri",
    "fokus kerja aja deh daripada ribut",
    "ga ngerti kenapa harus boikot",
    "hidup ya gitu aja, santai",
]
contra_texts = [
    "boikot tidak efektif, buang waktu saja",
    "kasihan karyawan lokal yang kena dampaknya",
    "produknya bagus, kenapa harus boikot",
    "boikot cuma trend sesaat, nanti juga beli lagi",
    "jangan ikut-ikutan tanpa pikir panjang",
    "ekonomi bisa terganggu kalau boikot massal",
    "karyawan Indonesia yang rugi, bukan perusahaan asing",
    "saya tetap beli karena kualitasnya terbaik",
    "boikot hanya merugikan diri sendiri",
    "lebih baik support UMKM daripada sibuk boikot",
]
text_pools = [pro_texts, neutral_texts, contra_texts]

rows = []
for _ in range(N_TWEETS):
    author = random.choice(users)
    group = author["group"]

    # Pick text from author's group (80%) or random (20%)
    if random.random() < 0.8:
        text = random.choice(text_pools[group])
    else:
        text = random.choice(pro_texts + neutral_texts + contra_texts)

    # Determine interaction target (70% within group, 30% cross-group)
    if random.random() < 0.7:
        targets = [u for u in users if u["group"] == group and u["id"] != author["id"]]
    else:
        targets = [u for u in users if u["id"] != author["id"]]
    target = random.choice(targets) if targets else None

    # Randomly assign interaction type
    reply_to = None
    retweet_from = None
    mentioned = None
    itype = random.choices(["reply", "retweet", "mention", "none"], weights=[35, 25, 20, 20])[0]
    if target and itype == "reply":
        reply_to = target["id"]
    elif target and itype == "retweet":
        retweet_from = target["id"]
    elif target and itype == "mention":
        mentioned = f"(name,{target['username']})//(id,{target['id']})"

    rows.append({
        "from_id": author["id"],
        "from_username": author["username"],
        "cleaned_text": text,
        "reply_to_user_id": reply_to,
        "retweet_from_user_id": retweet_from,
        "mentioned": mentioned,
    })

df = pd.DataFrame(rows)

# Save
import os
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data.csv")
df.to_csv(outpath, index=False)
print(f"Generated {len(df)} rows, {len(df['from_id'].unique())} unique users -> {outpath}")
print(f"Interaction breakdown:")
print(f"  Replies:  {df['reply_to_user_id'].notna().sum()}")
print(f"  Retweets: {df['retweet_from_user_id'].notna().sum()}")
print(f"  Mentions: {df['mentioned'].notna().sum()}")
