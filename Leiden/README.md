# ECR 2.0 — Echo Chamber Ratio Pipeline

Pipeline untuk mengukur **echo chamber** di media sosial Indonesia menggunakan **Sentiment Analysis (Fine-tuned XLM-RoBERTa)**, **Community Detection (Leiden + Infomap)**, dan **ECR 2.0 Framework** dengan validasi statistik.

Pipeline ini menggabungkan:

* **Fine-tuned Sentiment Analysis** (XLM-RoBERTa)
* **Interaction Graph Construction** dari data Twitter/X
* **Community Detection** (Leiden + Infomap, dijalankan parallel)
* **Echo Chamber Ratio** dengan Bootstrap CI & Permutation Test
* **Homophily, Diffusion Bias, dan Community Quality Metrics**
* **10 Visualisasi otomatis** (matplotlib)

Proyek ini dikembangkan sebagai bagian dari **Tugas Akhir** di
**Teknik Informatika, Sekolah Teknik Elektro dan Informatika, Institut Teknologi Bandung (ITB).**

---

## Abstract

Media sosial telah menjadi ruang penting bagi masyarakat Indonesia untuk berinteraksi dan membentuk opini, namun pola komunikasi yang terbentuk seringkali menciptakan *echo chamber* — lingkungan di mana pengguna hanya terpapar informasi yang sejalan dengan pandangannya, sehingga memperkuat polarisasi.

Penelitian ini mengembangkan pipeline untuk mengukur echo chamber di percakapan media sosial Indonesia menggunakan pendekatan **Echo Chamber Ratio (ECR)** yang menggabungkan analisis sentimen, deteksi komunitas, dan validasi statistik. Pipeline terdiri dari lima tahapan utama:

1. **Preprocessing & Sentiment Analysis** — Teks dibersihkan lalu dianalisis menggunakan model XLM-RoBERTa yang di-fine-tune pada data Indonesia, menghasilkan probabilitas sentimen 3-kelas (negatif, netral, positif) per teks.

2. **Konstruksi Graf Interaksi** — Interaksi antar pengguna (reply, retweet, mention) dibangun menjadi graf berarah berbobot, di mana setiap node adalah username dan setiap edge merepresentasikan interaksi.

3. **Estimasi User Leaning & Deteksi Komunitas** — Posisi (leaning) setiap pengguna diestimasi dari rata-rata probabilitas sentimen, kemudian komunitas dideteksi menggunakan dua algoritma secara parallel: **Leiden** (optimasi modularitas) dan **Infomap** (optimasi information flow).

4. **Perhitungan ECR & Validasi Statistik** — Echo Chamber Ratio dihitung dari perbandingan *intra-community agreement* (kesamaan pendapat dalam komunitas) vs *inter-community agreement* (kesamaan pendapat antar komunitas). Hasilnya divalidasi dengan:
   - **Null model threshold** — perbandingan terhadap graf random
   - **Bootstrap 95% Confidence Interval** — stabilitas estimasi ECR
   - **Permutation test (p-value)** — signifikansi statistik
   - **Community quality metrics** — Modularity, Coverage, Conductance, NMI

5. **Analisis Lanjutan & Visualisasi** — Homophily (korelasi leaning user dengan tetangganya), simulasi diffusion bias (Independent Cascade), klasifikasi stance per komunitas, dan 10 visualisasi otomatis.

Hasil menunjukkan bahwa pipeline mampu mendeteksi echo chamber secara kuantitatif dan memberikan bukti statistik yang kuat mengenai ada tidaknya echo chamber pada suatu topik percakapan.

---

## Tahapan Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Dataset CSV                            │
│         (from_id, from_username, cleaned_text,                  │
│          reply_to_user_id, retweet_from_user_id, mentioned)     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Load Data                                              │
│  • Baca CSV, validasi kolom                                     │
│  • Bangun username_map (user_id → username)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Sentiment Inference                                    │
│  • Fine-tuned XLM-RoBERTa (finetuned_sentiment/)               │
│  • Output: p_neg, p_neu, p_pos per teks                        │
│  • Auto-cache: hasil disimpan agar tidak perlu ulang            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Build Interaction Graph                                │
│  • Reply edges + Retweet edges + Mention edges                  │
│  • Node = username, Edge = interaksi (directed, weighted)       │
│  • Filter: hanya user yang punya konten                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Estimate User Leaning                                  │
│  • Agregasi probabilitas sentimen per user                      │
│  • lean_scalar = f(p_pos, p_neg, topic_polarity)                │
│  • Rentang: -1 (contra) sampai +1 (pro)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Calibration (opsional)                                 │
│  • Isotonic regression atau Temperature scaling                 │
│  • Memperbaiki probabilitas yang overconfident/underconfident    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: Community Detection (Parallel)                         │
│  ┌──────────────┐    ┌──────────────┐                           │
│  │   LEIDEN      │    │   INFOMAP    │                           │
│  │  (modularity) │    │ (info flow)  │                           │
│  └──────┬───────┘    └──────┬───────┘                           │
│         └────────┬──────────┘                                   │
│                  ▼                                               │
│  STEP 6b: Quality Validation                                    │
│  • Modularity, Coverage, Performance, Conductance               │
│  • NMI (agreement Leiden ↔ Infomap)                             │
│  • Verdict: RELIABLE / PERLU DICEK                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7: Compute ECR 2.0                                        │
│  • intra = rata-rata agreement DALAM komunitas                  │
│  • inter = rata-rata agreement ANTAR komunitas                  │
│  • ratio = intra / inter (< 1 = echo chamber)                  │
│  • Dihitung untuk Leiden DAN Infomap                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 8: ECR Threshold (Null Model)                             │
│  • Generate N graf random (shuffle edges)                       │
│  • Hitung ECR pada setiap graf random                           │
│  • Threshold = rata-rata ECR dari graf random                   │
│                                                                 │
│  STEP 8b: Statistical Validation                                │
│  • Bootstrap 95% CI (resample users 100x)                       │
│  • Permutation test p-value (shuffle leaning 100x)              │
│  • Verdict: ECHO CHAMBER DETECTED / NO ECHO CHAMBER             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 9-12: Analisis Lanjutan                                   │
│  • Homophily (Pearson r: user leaning ↔ neighbor leaning)       │
│  • Diffusion Bias (simulasi Independent Cascade)                │
│  • Community Classification (pro/neutral/contra per komunitas)  │
│  • Diffusion Metrics (slope, R², bias ratio, Cohen's d)         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 13-14: Output                                             │
│  • Simpan CSV, TXT ke output directory                          │
│  • Generate 10 visualisasi PNG                                  │
│  • Tampilkan di GUI browser                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT                                        │
│  • comparison_summary.txt (Leiden vs Infomap)                   │
│  • users_leaning.csv, communities.csv                           │
│  • ecr_metrics.txt (ECR + quality + validation)                 │
│  • 10 visualisasi PNG                                           │
│  • Verdict: echo chamber / bukan                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Instalasi

### 1. Clone & Setup Environment

```bash
cd ECR-2.0/Leiden
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Atau install manual:

```bash
pip install numpy pandas networkx nicegui matplotlib tqdm igraph leidenalg infomap scikit-learn scipy transformers torch
```

### Requirements Lengkap

| Package | Versi Min. | Fungsi |
|---------|-----------|--------|
| `numpy` | >= 1.24 | Operasi numerik |
| `pandas` | >= 2.0 | Manipulasi data (DataFrame, CSV) |
| `networkx` | >= 3.0 | Konstruksi & analisis graf |
| `nicegui` | >= 1.4 | Web GUI framework |
| `matplotlib` | >= 3.7 | Visualisasi (10 chart) |
| `tqdm` | >= 4.65 | Progress bar |
| `igraph` | >= 0.10 | Graf high-performance (untuk Leiden) |
| `leidenalg` | >= 0.10 | Algoritma Leiden community detection |
| `infomap` | >= 2.7 | Algoritma Infomap community detection |
| `scikit-learn` | >= 1.3 | Kalibrasi, NMI, metrik evaluasi |
| `scipy` | >= 1.10 | Optimisasi & statistik |
| `transformers` | >= 4.30 | Fine-tuned model inference (HuggingFace) |
| `torch` | >= 2.0 | Backend untuk transformers |

---

## Cara Menjalankan

### GUI Mode (Recommended)

```bash
cd Leiden
python main_gui.py
```

Buka browser di **http://localhost:8080**

**Langkah-langkah:**

1. Masukkan path dataset CSV (e.g. `total_data_cleaned_mcd.csv`)
2. Konfigurasi parameter (atau gunakan default)
3. Klik **Run Pipeline** — hasil muncul di browser
4. Untuk demo/presentasi: klik **Load Previous Results** untuk menampilkan hasil pipeline sebelumnya secara instan

### CLI Mode (via Notebook)

Lihat notebook di folder `code_boikot/`, `code_indonesia_gelap/`, atau `code_vaksin/` untuk contoh pipeline lengkap via Jupyter.

### Test Cepat

```bash
python generate_test_data.py    # Generate test_data.csv (800 rows, 150 users)
python main_gui.py              # Jalankan GUI, masukkan path test_data.csv
```

---

## Struktur Folder

```
Leiden/
│
├── main_gui.py                  # GUI utama (NiceGUI, port 8080)
├── ecr2_pipeline.py             # Library pipeline ECR (semua fungsi core)
├── visualizations.py            # Modul visualisasi (10 chart matplotlib)
├── generate_test_data.py        # Generator test data sintetis
│
├── finetuned_sentiment/         # Fine-tuned XLM-RoBERTa model
│   ├── config.json              # Model config (id2label mapping)
│   ├── model.safetensors        # Model weights
│   ├── tokenizer.json           # Tokenizer
│   ├── training_config.json     # Training hyperparameters
│   └── checkpoint-*/            # Training checkpoints
│
├── data/                        # Dataset mentah
│   ├── IndSight_Isu_Nasional/   # Data Indonesia Gelap
│   ├── McDonald's_Indonesia/    # Data Boikot McDonald's
│   └── Vaksinasi/               # Data Vaksinasi
│
├── total_data_cleaned_*.csv     # Dataset yang sudah dibersihkan (input pipeline)
│   ├── total_data_cleaned_mcd.csv           # Boikot (~534 MB, ~515K rows)
│   ├── total_data_cleaned_indonesia_gelap.csv
│   └── total_data_cleaned_vaksin.csv
│
├── test_data.csv                # Data test kecil (800 rows)
│
├── code_boikot/                 # Notebook pipeline topik Boikot
│   └── ECR2_Full_Pipeline.ipynb
├── code_indonesia_gelap/        # Notebook pipeline topik Indonesia Gelap
├── code_vaksin/                 # Notebook pipeline topik Vaksinasi
│
├── hasil_boikot/                # Output hasil analisis Boikot
│   ├── 700K Data/               # Hasil dari 700K data (10 visualisasi + CSV)
│   └── 10K Data Indobert/       # Hasil perbandingan IndoBERT
├── hasil_indonesia_gelap/       # Output hasil analisis Indonesia Gelap
├── hasil_vaksin/                # Output hasil analisis Vaksinasi
│
├── pipeline_out/                # Output default dari GUI pipeline
│   ├── visualizations/          # 10 chart PNG
│   ├── leiden/                  # Hasil Leiden (communities, ECR, metrics)
│   ├── infomap/                 # Hasil Infomap
│   ├── users_leaning.csv        # User leaning scores
│   ├── comparison_summary.txt   # Perbandingan Leiden vs Infomap
│   └── *_sentiment_cache.csv    # Cache hasil inferensi model
│
├── sentiment_finetuning.ipynb   # Notebook fine-tuning model sentimen
├── preprocessing.ipynb          # Notebook preprocessing data
├── .env                         # Environment variables
└── README.md                    # File ini
```

---

## File-File Penting

### `main_gui.py` — GUI Pipeline

Web interface (NiceGUI) yang mengorkestrasi seluruh pipeline. Fitur:

- Input path CSV dataset
- Konfigurasi semua parameter dengan tooltip (i) info
- Sentiment inference dengan fine-tuned model (auto-cache)
- Leiden + Infomap dijalankan parallel
- Community quality metrics (Modularity, Coverage, Conductance, NMI)
- Bootstrap CI + Permutation Test untuk validasi ECR
- 10 visualisasi otomatis
- Tombol "Load Previous Results" untuk demo/presentasi

### `ecr2_pipeline.py` — Core Library

Berisi semua fungsi pipeline:

| Fungsi | Deskripsi |
|--------|-----------|
| `build_graph()` | Membangun graf interaksi dari DataFrame |
| `estimate_user_leaning()` | Menghitung leaning user dari probabilitas sentimen |
| `calibrate_probabilities()` | Kalibrasi probabilitas (isotonic/temperature) |
| `detect_communities()` | Deteksi komunitas (Leiden/Infomap/Greedy fallback) |
| `compute_ecr2()` | Menghitung Echo Chamber Ratio |
| `compute_homophily()` | Mengukur homophily (Pearson r) |
| `simulate_diffusion_bias()` | Simulasi Independent Cascade (IC) |
| `summarize_diffusion_bias()` | Merangkum hasil difusi |
| `estimate_ecr_threshold()` | Estimasi threshold ECR dari null model |
| `classify_echo_chamber()` | Klasifikasi echo chamber / non-echo chamber |
| `classify_communities_detailed()` | Klasifikasi komunitas (pro/neutral/contra) |
| `compute_diffusion_bias_metrics()` | Metrik bias difusi per komunitas |
| `sentiment_indobert()` | Inferensi sentimen via HuggingFace model |

### `visualizations.py` — 10 Visualisasi

| # | File Output | Deskripsi |
|---|-------------|-----------|
| 01 | `01_calibration_impact.png` | Reliability diagrams (4 metode kalibrasi) |
| 01b | `01b_calibration_per_confidence.png` | ECE per confidence threshold |
| 02 | `02_user_leaning.png` | Distribusi leaning + confidence + scatter |
| 03 | `03_community_sizes.png` | Ukuran komunitas (Leiden vs Infomap) |
| 04 | `04_ecr_results.png` | ECR metrics + homophily scatter |
| 05 | `05_diffusion_bias.png` | Scatter regression + bias per komunitas |
| 06 | `06_pro_contra.png` | Analisis stance + top users + komunitas |
| 07 | `07_shifts.png` | Pergeseran sentimen (early vs late) |
| 08 | `08_network.png` | Network graph top-degree users |
| 09 | `09_degrees.png` | Degree distribution (total, in, out) |
| 10 | `10_sentiment_sample.png` | Tabel 10 contoh hasil sentimen |

### `finetuned_sentiment/` — Fine-tuned Model

Model XLM-RoBERTa (`cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`) yang sudah di-fine-tune untuk data ini. Label: `negative`, `neutral`, `positive`.

---

## Format Dataset Input

Dataset CSV harus mengandung kolom berikut:

| Kolom | Wajib | Deskripsi |
|-------|-------|-----------|
| `from_id` | Ya | User ID pengirim |
| `from_username` | Ya | Username pengirim (digunakan sebagai node label) |
| `cleaned_text` | Ya | Teks yang sudah dibersihkan (input model sentimen) |
| `reply_to_user_id` | Min. 1 | User ID target reply |
| `retweet_from_user_id` | Min. 1 | User ID sumber retweet |
| `mentioned` | Min. 1 | Mentioned users: `(name,USERNAME)//(id,USERID)` |

Minimal satu dari `reply_to_user_id`, `retweet_from_user_id`, atau `mentioned` harus ada untuk membangun graf interaksi.

---

## Pipeline Steps

Pipeline GUI menjalankan 14 langkah:

1. **Load Data** — Baca CSV, validasi kolom, build username map
2. **Sentiment Inference** — Fine-tuned model inference (auto-cache setelah run pertama)
3. **Build Graph** — Konstruksi graf interaksi (reply, retweet, mention edges)
4. **User Leaning** — Estimasi leaning per user dari probabilitas sentimen
5. **Calibration** — Kalibrasi probabilitas (opsional, isotonic/temperature)
6. **Community Detection** — Leiden + Infomap (parallel)
6b. **Quality Validation** — Modularity, Coverage, Conductance, NMI
7. **ECR 2.0** — Hitung intra/inter agreement dan rasio ECR
8. **ECR Threshold** — Estimasi threshold dari null model random
8b. **Statistical Validation** — Bootstrap 95% CI + Permutation Test (p-value)
9. **Homophily** — Pearson correlation (user leaning vs neighbor leaning)
10. **Diffusion Bias** — Simulasi Independent Cascade (IC)
11. **Community Classification** — Klasifikasi stance per komunitas
12. **Diffusion Metrics** — Slope, R², bias ratio, Cohen's d
13. **Save Outputs** — CSV, TXT ke output directory
14. **Visualizations** — Generate 10 chart PNG

---

## Metrik Validasi ECR

Pipeline menggunakan multiple evidence untuk memvalidasi hasil ECR:

| Metrik | Arti | Threshold |
|--------|------|-----------|
| **ECR Ratio** | Rasio intra vs inter community agreement | < threshold = echo chamber |
| **Null Model Threshold** | ECR dari graf random | Baseline perbandingan |
| **Bootstrap 95% CI** | Interval kepercayaan ECR | Seluruh CI < threshold = yakin |
| **Permutation p-value** | Signifikansi statistik | p < 0.05 = signifikan |
| **Modularity** | Kualitas partisi komunitas | > 0.3 = meaningful |
| **Coverage** | Fraksi intra-community edges | Tinggi = baik |
| **Conductance** | "Kebocoran" batas komunitas | < 0.5 = well-separated |
| **NMI** | Agreement Leiden vs Infomap | > 0.5 = konsisten |
| **Homophily (Pearson r)** | Korelasi leaning user-neighbor | > 0.3 = homophily ada |

---

## Output

Setelah pipeline selesai, output tersimpan di folder `pipeline_out/` (default):

```
pipeline_out/
├── users_leaning.csv            # User + leaning score
├── comparison_summary.txt       # Perbandingan lengkap Leiden vs Infomap
├── diffusion_ic.csv             # Hasil simulasi IC
├── diffusion_summary.csv        # Rangkuman difusi
├── *_sentiment_cache.csv        # Cache inferensi model
├── leiden/
│   ├── communities.csv          # User -> community mapping
│   ├── community_classification.csv  # Stance per komunitas
│   ├── ecr_metrics.txt          # ECR + quality + validation
│   └── diffusion_bias_metrics.csv
├── infomap/
│   └── (sama seperti leiden/)
└── visualizations/
    ├── 01_calibration_impact.png
    ├── 01b_calibration_per_confidence.png
    ├── 02_user_leaning.png
    ├── 03_community_sizes.png
    ├── 04_ecr_results.png
    ├── 05_diffusion_bias.png
    ├── 06_pro_contra.png
    ├── 07_shifts.png
    ├── 08_network.png
    ├── 09_degrees.png
    └── 10_sentiment_sample.png
```

---

## Author

* **Amalia Putri**

---

## Disclaimer

Proyek ini dibuat sebagai bagian dari **Tugas Akhir** di
**Teknik Informatika, Sekolah Teknik Elektro dan Informatika, Institut Teknologi Bandung (ITB), 2025.**
