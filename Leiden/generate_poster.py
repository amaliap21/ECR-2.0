"""
Generate A0 landscape research poster for ECR 2.0 Pipeline.
Output: poster_ecr2.png (high-res) + poster_ecr2.pdf

Style: academic poster similar to ITB conference poster format.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import os
from pathlib import Path
import textwrap

# ── Configuration ──────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent
POSTER_W, POSTER_H = 42, 30  # inches (A0 landscape-ish)
DPI = 150

# Colors
C_HEADER = '#1a237e'     # dark blue header
C_SECTION = '#283593'    # section title bg
C_ACCENT = '#42a5f5'     # accent blue
C_BG = '#fafafa'         # poster bg
C_WHITE = '#ffffff'
C_TEXT = '#212121'
C_LIGHT = '#e3f2fd'      # light blue bg for boxes
C_GREEN = '#2e7d32'
C_RED = '#c62828'
C_ORANGE = '#ef6c00'
C_BORDER = '#90caf9'

# Visualization images (try loading from hasil_boikot/700K Data)
VIZ_DIR = Path(__file__).parent / "hasil_boikot" / "700K Data"
VIZ_DIR2 = Path(__file__).parent / "pipeline_out" / "visualizations"


def wrap(text, width=55):
    return "\n".join(textwrap.wrap(text, width=width))


def add_section_header(ax, title, x, y, w, h, fontsize=16):
    """Draw a colored section header box."""
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01",
        facecolor=C_SECTION, edgecolor='none',
        transform=ax.transAxes
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, title,
            transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold',
            color=C_WHITE, ha='center', va='center',
            family='sans-serif')


def add_content_box(ax, x, y, w, h, text, fontsize=10, align='left', color=C_TEXT, bg=None, border=None):
    """Draw text in a box area."""
    if bg:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.008",
            facecolor=bg, edgecolor=border or 'none',
            linewidth=1.5 if border else 0,
            transform=ax.transAxes
        )
        ax.add_patch(rect)
    ha = 'left' if align == 'left' else 'center'
    tx = x + 0.008 if align == 'left' else x + w/2
    ax.text(tx, y + h - 0.008, text,
            transform=ax.transAxes,
            fontsize=fontsize, color=color,
            ha=ha, va='top',
            family='sans-serif',
            linespacing=1.4)


def load_viz(name):
    """Try loading a visualization image."""
    for d in [VIZ_DIR, VIZ_DIR2]:
        p = d / name
        if p.exists():
            return plt.imread(str(p))
    return None


def build_poster():
    fig = plt.figure(figsize=(POSTER_W, POSTER_H), facecolor=C_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # ══════════════════════════════════════════════════════════════════════
    # HEADER BAR
    # ══════════════════════════════════════════════════════════════════════
    header = mpatches.FancyBboxPatch(
        (0, 0.88), 1, 0.12,
        boxstyle="square,pad=0",
        facecolor=C_HEADER, edgecolor='none',
        transform=ax.transAxes
    )
    ax.add_patch(header)

    # Title
    ax.text(0.5, 0.955,
            "Pengukuran Echo Chamber di Media Sosial Indonesia\n"
            "Menggunakan Analisis Sentimen dan Deteksi Komunitas\n"
            "dengan Framework Echo Chamber Ratio (ECR) 2.0",
            transform=ax.transAxes,
            fontsize=24, fontweight='bold', color=C_WHITE,
            ha='center', va='center', family='sans-serif',
            linespacing=1.3)

    # Author
    ax.text(0.5, 0.893,
            "Amalia Putri  |  Teknik Informatika, Sekolah Teknik Elektro dan Informatika, Institut Teknologi Bandung (ITB)",
            transform=ax.transAxes,
            fontsize=13, color='#bbdefb',
            ha='center', va='center', family='sans-serif')

    # ══════════════════════════════════════════════════════════════════════
    # COLUMN 1 (left): Abstract, Latar Belakang, Metodologi
    # ══════════════════════════════════════════════════════════════════════
    col1_x = 0.015
    col_w = 0.235

    # ── ABSTRACT ──
    y = 0.855
    add_section_header(ax, "ABSTRACT", col1_x, y, col_w, 0.022)
    abstract_text = (
        "Media sosial telah menjadi ruang penting bagi masyarakat Indonesia "
        "untuk berinteraksi dan membentuk opini. Namun, pola komunikasi yang "
        "terbentuk seringkali menciptakan echo chamber — lingkungan di mana "
        "pengguna hanya terpapar informasi yang sejalan dengan pandangannya.\n\n"
        "Penelitian ini mengembangkan pipeline otomatis untuk mengukur echo "
        "chamber menggunakan pendekatan Echo Chamber Ratio (ECR) 2.0 yang "
        "menggabungkan:\n"
        "• Fine-tuned Sentiment Analysis (XLM-RoBERTa)\n"
        "• Dual Community Detection (Leiden + Infomap)\n"
        "• Validasi Statistik (Bootstrap CI, Permutation Test)\n"
        "• Community Quality Metrics & Homophily\n\n"
        "Pipeline diimplementasikan sebagai web GUI (NiceGUI) dan diuji pada "
        "tiga dataset percakapan di media sosial Indonesia."
    )
    add_content_box(ax, col1_x, y - 0.195, col_w, 0.19, abstract_text,
                    fontsize=9.5, bg=C_LIGHT, border=C_BORDER)

    # ── LATAR BELAKANG ──
    y2 = y - 0.22
    add_section_header(ax, "LATAR BELAKANG", col1_x, y2, col_w, 0.022)
    bg_text = (
        "Permasalahan:\n"
        "• 73% populasi Indonesia aktif di media sosial\n"
        "• Echo chamber memperkuat polarisasi & misinformasi\n"
        "• Studi echo chamber di Indonesia masih terbatas\n"
        "• Pendekatan manual tidak scalable untuk data besar\n\n"
        "Pendekatan Sebelumnya:\n"
        "• Network-based: hanya struktur, abaikan konten\n"
        "• Content-based: capture sentimen, miss hubungan\n"
        "• Hybrid (Amendola et al.): ABSA + GDM consensus\n\n"
        "Kontribusi Penelitian Ini:\n"
        "• Pipeline otomatis end-to-end\n"
        "• Fine-tuned model untuk Bahasa Indonesia\n"
        "• Dual community detection (Leiden + Infomap)\n"
        "• Multiple validation metrics\n"
        "• Web GUI untuk analisis interaktif"
    )
    add_content_box(ax, col1_x, y2 - 0.265, col_w, 0.26, bg_text,
                    fontsize=9, bg=C_LIGHT, border=C_BORDER)

    # ── METODOLOGI ──
    y3 = y2 - 0.29
    add_section_header(ax, "METODOLOGI", col1_x, y3, col_w, 0.022)
    method_text = (
        "Pipeline terdiri dari 5 tahapan utama:\n\n"
        "1. Preprocessing & Sentiment Analysis\n"
        "   Model: XLM-RoBERTa (fine-tuned)\n"
        "   Output: p_neg, p_neu, p_pos per tweet\n\n"
        "2. Konstruksi Graf Interaksi\n"
        "   Edge: reply, retweet, mention (directed)\n"
        "   Node: username pengguna\n\n"
        "3. User Leaning & Community Detection\n"
        "   Leaning: avg probability → [-1, +1]\n"
        "   Algoritma: Leiden + Infomap (parallel)\n\n"
        "4. ECR 2.0 & Validasi Statistik\n"
        "   ECR = intra_agreement / inter_agreement\n"
        "   Null model, Bootstrap CI, Permutation test\n\n"
        "5. Analisis Lanjutan & Visualisasi\n"
        "   Homophily, Diffusion Bias (IC), Stance\n"
        "   10 visualisasi otomatis"
    )
    add_content_box(ax, col1_x, y3 - 0.30, col_w, 0.295, method_text,
                    fontsize=9, bg=C_LIGHT, border=C_BORDER)

    # ══════════════════════════════════════════════════════════════════════
    # COLUMN 2 (center-left): Pipeline Diagram + Formulas + Datasets
    # ══════════════════════════════════════════════════════════════════════
    col2_x = 0.265
    col2_w = 0.24

    # ── PIPELINE FLOWCHART ──
    y = 0.855
    add_section_header(ax, "PIPELINE OVERVIEW", col2_x, y, col2_w, 0.022)

    # Draw simplified pipeline as connected boxes
    steps = [
        ("Input CSV", "from_id, username, text,\nreply/retweet/mention"),
        ("Sentiment Analysis", "XLM-RoBERTa fine-tuned\n→ p_neg, p_neu, p_pos"),
        ("Graf Interaksi", "Directed weighted graph\nNode=user, Edge=interaction"),
        ("User Leaning", "Agregasi sentimen per user\nlean ∈ [-1, +1]"),
        ("Community Detection", "Leiden (modularity)\nInfomap (info flow)"),
        ("ECR 2.0", "intra / inter agreement\n+ null model threshold"),
        ("Validasi Statistik", "Bootstrap 95% CI\nPermutation test p-value"),
        ("Output & Visualisasi", "10 chart + metrics\n+ verdict"),
    ]

    box_h = 0.032
    gap = 0.004
    start_y = y - 0.04
    bw = col2_w - 0.02
    bx = col2_x + 0.01

    step_colors = [C_ACCENT, '#66bb6a', '#66bb6a', '#ffa726',
                   '#ab47bc', C_RED, C_RED, C_ACCENT]

    for i, (title, desc) in enumerate(steps):
        by = start_y - i * (box_h + gap)
        rect = mpatches.FancyBboxPatch(
            (bx, by), bw, box_h,
            boxstyle="round,pad=0.004",
            facecolor=step_colors[i], edgecolor='white',
            linewidth=1.5, alpha=0.85,
            transform=ax.transAxes
        )
        ax.add_patch(rect)
        ax.text(bx + 0.005, by + box_h/2 + 0.005,
                f"Step {i+1}: {title}", transform=ax.transAxes,
                fontsize=8.5, fontweight='bold', color=C_WHITE,
                va='center', family='sans-serif')
        ax.text(bx + bw - 0.005, by + box_h/2 - 0.003,
                desc, transform=ax.transAxes,
                fontsize=7, color=C_WHITE, alpha=0.9,
                ha='right', va='center', family='sans-serif')

        # Arrow
        if i < len(steps) - 1:
            arrow_y = by - gap/2
            ax.annotate('', xy=(bx + bw/2, arrow_y - 0.001),
                        xytext=(bx + bw/2, arrow_y + 0.003),
                        transform=ax.transAxes,
                        arrowprops=dict(arrowstyle='->', color=C_TEXT,
                                        lw=1.5))

    # ── FORMULA ECR ──
    y_form = start_y - len(steps) * (box_h + gap) - 0.015
    add_section_header(ax, "FORMULA ECR 2.0", col2_x, y_form, col2_w, 0.022)
    formula_text = (
        "Echo Chamber Ratio (ECR):\n\n"
        "        E[intra-community agreement]\n"
        "ECR = ─────────────────────────────────\n"
        "        E[inter-community agreement]\n\n"
        "• intra = avg similarity antara user\n"
        "  dalam komunitas yang SAMA\n"
        "• inter = avg similarity antara user\n"
        "  dari komunitas BERBEDA\n\n"
        "Interpretasi:\n"
        "  ECR < threshold → Echo Chamber\n"
        "  ECR ≥ threshold → Non-Echo Chamber\n\n"
        "Threshold diestimasi dari null model\n"
        "(N graf random, shuffle edges)"
    )
    add_content_box(ax, col2_x, y_form - 0.23, col2_w, 0.225, formula_text,
                    fontsize=9, bg=C_LIGHT, border=C_BORDER)

    # ── DATASET ──
    y_ds = y_form - 0.255
    add_section_header(ax, "DATASET", col2_x, y_ds, col2_w, 0.022)
    ds_text = (
        "Tiga dataset percakapan Twitter/X Indonesia:\n\n"
        "Dataset               Rows        Users\n"
        "─────────────────────────────────────\n"
        "Boikot McDonald's    ~515K       ~168K\n"
        "Indonesia Gelap       varies      varies\n"
        "Vaksinasi             varies      varies\n\n"
        "Format input: CSV dengan kolom\n"
        "from_id, from_username, cleaned_text,\n"
        "reply_to_user_id, retweet_from_user_id,\n"
        "mentioned"
    )
    add_content_box(ax, col2_x, y_ds - 0.175, col2_w, 0.17, ds_text,
                    fontsize=9, bg=C_LIGHT, border=C_BORDER)

    # ══════════════════════════════════════════════════════════════════════
    # COLUMN 3 (center-right): Results / Visualizations
    # ══════════════════════════════════════════════════════════════════════
    col3_x = 0.52
    col3_w = 0.24

    y = 0.855
    add_section_header(ax, "HASIL & VISUALISASI", col3_x, y, col3_w, 0.022)

    # ── ECR Results Table ──
    result_text = (
        "Hasil ECR — Boikot McDonald's (515K tweets):\n\n"
        "Metrik                  Leiden    Infomap\n"
        "───────────────────────────────────────\n"
        "Komunitas                 2138       3663\n"
        "Intra agreement          0.828      0.831\n"
        "Inter agreement          0.849      0.848\n"
        "ECR 2.0 ratio            0.976      0.981\n"
        "Threshold                1.000      1.000\n"
        "Klasifikasi        Non-Echo  Non-Echo\n\n"
        "Community Quality:\n"
        "Modularity              0.634      0.640\n"
        "Coverage                0.619      0.712\n"
        "Conductance (mean)      0.031      0.052\n"
        "NMI (Leiden↔Infomap)      0.679\n"
        "Homophily (Pearson r)     0.490"
    )
    add_content_box(ax, col3_x, y - 0.255, col3_w, 0.25, result_text,
                    fontsize=9, bg=C_LIGHT, border=C_BORDER)

    # Load and show 2 key visualizations
    viz_y = y - 0.28

    # Try loading ECR results chart
    img_ecr = load_viz("04_ecr_results.png")
    if img_ecr is not None:
        ax_img1 = fig.add_axes([col3_x + 0.005, viz_y - 0.22, col3_w - 0.01, 0.2])
        ax_img1.imshow(img_ecr)
        ax_img1.set_axis_off()
        ax_img1.set_title("ECR Results & Homophily", fontsize=9, fontweight='bold', pad=2)
        next_y = viz_y - 0.24
    else:
        next_y = viz_y - 0.01

    # Try loading community sizes
    img_comm = load_viz("03_community_sizes.png")
    if img_comm is not None:
        ax_img2 = fig.add_axes([col3_x + 0.005, next_y - 0.20, col3_w - 0.01, 0.18])
        ax_img2.imshow(img_comm)
        ax_img2.set_axis_off()
        ax_img2.set_title("Community Sizes (Leiden vs Infomap)", fontsize=9, fontweight='bold', pad=2)
        next_y = next_y - 0.22

    # Try loading network
    img_net = load_viz("08_network.png")
    if img_net is not None:
        ax_img3 = fig.add_axes([col3_x + 0.005, next_y - 0.18, col3_w - 0.01, 0.16])
        ax_img3.imshow(img_net)
        ax_img3.set_axis_off()
        ax_img3.set_title("Network Graph (Top Degree Users)", fontsize=9, fontweight='bold', pad=2)

    # ══════════════════════════════════════════════════════════════════════
    # COLUMN 4 (right): Validation, Analysis, Conclusion
    # ══════════════════════════════════════════════════════════════════════
    col4_x = 0.775
    col4_w = 0.21

    y = 0.855
    add_section_header(ax, "VALIDASI STATISTIK", col4_x, y, col4_w, 0.022)
    val_text = (
        "Metrik Validasi ECR:\n\n"
        "Metrik            Threshold   Arti\n"
        "──────────────────────────────────\n"
        "ECR Ratio         < threshold  Echo chamber\n"
        "Null Model        baseline     Perbandingan\n"
        "Bootstrap CI      95%          Stabilitas\n"
        "Permutation       p < 0.05     Signifikansi\n"
        "Modularity        > 0.3        Meaningful\n"
        "Coverage          tinggi       Baik\n"
        "Conductance       < 0.5        Separated\n"
        "NMI               > 0.5        Konsisten\n"
        "Homophily         > 0.3        Ada homophily\n\n"
        "Verdict Boikot McDonald's:\n"
        "  ✓ Modularity > 0.3 → RELIABLE\n"
        "  ✓ NMI = 0.679 → Leiden & Infomap\n"
        "    menghasilkan komunitas yang konsisten\n"
        "  ✓ ECR > threshold → Non-Echo Chamber\n"
        "  ✓ Homophily r=0.49 → moderate homophily"
    )
    add_content_box(ax, col4_x, y - 0.30, col4_w, 0.295, val_text,
                    fontsize=8.5, bg=C_LIGHT, border=C_BORDER)

    # ── KEY INSIGHTS ──
    y_ki = y - 0.325
    add_section_header(ax, "KEY INSIGHTS", col4_x, y_ki, col4_w, 0.022)
    insights_text = (
        "• Boikot McDonald's menunjukkan Non-Echo\n"
        "  Chamber (ECR ≈ 0.98, > threshold 1.0)\n"
        "• Komunitas memiliki modularity tinggi\n"
        "  (>0.63) → pembagian komunitas jelas\n"
        "• Homophily moderat (r=0.49) → user\n"
        "  cenderung berinteraksi dengan user\n"
        "  yang se-pandangan, tapi tidak ekstrem\n"
        "• Leiden & Infomap konsisten (NMI=0.68)\n"
        "  → hasil deteksi komunitas reliable\n"
        "• Diffusion bias slope positif (0.30)\n"
        "  → informasi sedikit bias ke arah\n"
        "  user yang sepandangan"
    )
    add_content_box(ax, col4_x, y_ki - 0.19, col4_w, 0.185, insights_text,
                    fontsize=8.5, bg=C_LIGHT, border=C_BORDER)

    # ── KESIMPULAN ──
    y_conc = y_ki - 0.215
    add_section_header(ax, "KESIMPULAN", col4_x, y_conc, col4_w, 0.022)
    conclusion_text = (
        "1. Pipeline ECR 2.0 berhasil mengukur\n"
        "   echo chamber secara kuantitatif\n"
        "   dengan validasi statistik yang kuat.\n\n"
        "2. Fine-tuned XLM-RoBERTa efektif\n"
        "   untuk analisis sentimen bahasa\n"
        "   Indonesia informal (media sosial).\n\n"
        "3. Dual community detection (Leiden +\n"
        "   Infomap) memberikan cross-validation\n"
        "   yang meningkatkan reliabilitas.\n\n"
        "4. Web GUI memungkinkan analisis\n"
        "   interaktif tanpa coding, cocok untuk\n"
        "   peneliti dan policy-maker."
    )
    add_content_box(ax, col4_x, y_conc - 0.21, col4_w, 0.205, conclusion_text,
                    fontsize=8.5, bg=C_LIGHT, border=C_BORDER)

    # ── REFERENSI ──
    y_ref = y_conc - 0.235
    add_section_header(ax, "REFERENSI", col4_x, y_ref, col4_w, 0.022)
    ref_text = (
        "[1] Amendola et al. (2024). ABSA-GDM\n"
        "    for Echo Chamber Measurement.\n"
        "[2] Traag et al. (2019). From Louvain\n"
        "    to Leiden. Scientific Reports.\n"
        "[3] Rosvall & Bergstrom (2008).\n"
        "    Maps of random walks on complex\n"
        "    networks. PNAS.\n"
        "[4] Barbera (2015). How Social Media\n"
        "    Reduces Mass Political Polarization."
    )
    add_content_box(ax, col4_x, y_ref - 0.145, col4_w, 0.14, ref_text,
                    fontsize=8, bg=C_LIGHT, border=C_BORDER)

    # ── FOOTER ──
    footer = mpatches.FancyBboxPatch(
        (0, 0), 1, 0.025,
        boxstyle="square,pad=0",
        facecolor=C_HEADER, edgecolor='none',
        transform=ax.transAxes
    )
    ax.add_patch(footer)
    ax.text(0.5, 0.012,
            "Institut Teknologi Bandung — Teknik Informatika — STEI — 2025  |  "
            "Keywords: echo chamber, sentiment analysis, community detection, ECR, "
            "Leiden, Infomap, XLM-RoBERTa, media sosial Indonesia",
            transform=ax.transAxes,
            fontsize=9, color='#bbdefb',
            ha='center', va='center', family='sans-serif')

    # Save
    poster_png = OUT_DIR / "poster_ecr2.png"
    poster_pdf = OUT_DIR / "poster_ecr2.pdf"

    fig.savefig(str(poster_png), dpi=DPI, bbox_inches='tight',
                facecolor=C_BG, edgecolor='none')
    fig.savefig(str(poster_pdf), dpi=DPI, bbox_inches='tight',
                facecolor=C_BG, edgecolor='none')
    plt.close(fig)
    print(f"[OK] Poster saved: {poster_png}")
    print(f"[OK] Poster saved: {poster_pdf}")


if __name__ == "__main__":
    build_poster()
