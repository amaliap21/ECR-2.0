"""Runner terpisah: cuplikan kecil graf interaksi (out-degree deg_min..deg_max),
dengan arah panah + label tipe interaksi (retweet/reply/mention) pada garis.
Memproses topik vaksin DAN boikot. Tidak mengubah pipeline lama.
Jalankan: python plot_network_snippet_run.py
"""
import os
import re
import pandas as pd
import networkx as nx
import visualizations as viz

# --- konfigurasi ---
DEG_MIN, DEG_MAX, MAX_SEEDS = 4, 4, 1   # versi mini: 1 seed berderajat 4 (~5 orang)
FNAME = "08c_network_mini.png"

TOPICS = [
    # {"name": "vaksin", "csv": "total_data_cleaned_vaksin.csv",
    #  "lean": "pipeline_out/new/vaksin/users_leaning.csv",
    #  "out":  "pipeline_out/new/vaksin/visualizations"},
    # {"name": "boikot", "csv": "total_data_cleaned_mcd.csv",
    #  "lean": "pipeline_out/new/boikot/users_leaning.csv",
    #  "out":  "pipeline_out/new/boikot/visualizations"},
    {"name": "indonesia_gelap", "csv": "total_data_cleaned_indonesia_gelap.csv",
     "lean": "pipeline_out/new/indonesia_gelap/users_leaning.csv",
     "out":  "pipeline_out/new/indonesia_gelap/visualizations"},
    {"name": "korupsi", "csv": "total_data_cleaned_korupsi.csv",
     "lean": "pipeline_out/new/korupsi/users_leaning.csv",
     "out":  "pipeline_out/new/korupsi/visualizations"},
    {"name": "ijazah", "csv": "total_data_cleaned_ijazah.csv",
     "lean": "pipeline_out/new/ijazah/users_leaning.csv",
     "out":  "pipeline_out/new/ijazah/visualizations"},
    {"name": "mbg", "csv": "total_data_cleaned_mbg.csv",
     "lean": "pipeline_out/new/mbg/users_leaning.csv",
     "out":  "pipeline_out/new/mbg/visualizations"},
]

MENT_RE = re.compile(r"\(id,([^)]*)\)//\(name,([^)]*)\)")

def parse_mentioned(raw):
    if not isinstance(raw, str):
        return []
    return [(i.strip(), n.strip()) for i, n in MENT_RE.findall(raw)]

def norm_id(x):
    if not isinstance(x, str):
        return None
    x = x.strip()
    if x == "" or x.lower() == "nan":
        return None
    if "e" in x.lower() or x.endswith(".0"):
        try:
            return str(int(float(x)))
        except Exception:
            return None
    return x

def build_graph(df):
    id2name = {}
    if "from_id" in df and "from_username" in df:
        for fid, fnm in zip(df["from_id"], df["from_username"]):
            if isinstance(fid, str) and isinstance(fnm, str) and fnm.strip():
                id2name.setdefault(fid.strip(), fnm.strip())
    for raw in df.get("mentioned", []):
        for uid, nm in parse_mentioned(raw):
            if uid and nm:
                id2name.setdefault(uid, nm)

    G = nx.DiGraph()

    def add(s, d, t):
        if not s or not d or s == d:
            return
        if G.has_edge(s, d):
            G[s][d]["weight"] += 1
            G[s][d]["types"].add(t)
        else:
            G.add_edge(s, d, weight=1, types={t})

    for _, row in df.iterrows():
        su = row.get("from_username")
        if not isinstance(su, str) or not su.strip():
            continue
        su = su.strip()
        rid = norm_id(row.get("reply_to_user_id"))
        if rid and rid in id2name:
            add(su, id2name[rid], "reply")
        tid = norm_id(row.get("retweet_from_user_id"))
        if tid and tid in id2name:
            add(su, id2name[tid], "retweet")
        for uid, nm in parse_mentioned(row.get("mentioned")):
            add(su, id2name.get(uid, nm), "mention")
    return G

for cfg in TOPICS:
    if not os.path.exists(cfg["csv"]) or not os.path.exists(cfg["lean"]):
        print(f"SKIP {cfg['name']} (berkas tidak ditemukan)")
        continue
    os.makedirs(cfg["out"], exist_ok=True)
    df = pd.read_csv(cfg["csv"], low_memory=False, dtype=str)
    G = build_graph(df)
    print(f"[{cfg['name']}] Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    dfu = pd.read_csv(cfg["lean"])
    dfu = dfu.rename(columns={dfu.columns[0]: "user"})
    od = dict(G.out_degree())
    n_seed = sum(1 for d in od.values() if DEG_MIN <= d <= DEG_MAX)
    print(f"[{cfg['name']}] pengguna out-degree {DEG_MIN}-{DEG_MAX}: {n_seed:,} (diambil {MAX_SEEDS})")
    path = viz.plot_network_snippet(G, dfu, cfg["out"], deg_min=DEG_MIN, deg_max=DEG_MAX,
                                    max_seeds=MAX_SEEDS, fname=FNAME)
    print(f"[{cfg['name']}] SAVED: {os.path.abspath(path)}")
