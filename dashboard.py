import os
import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.stats import zscore
from collections import Counter
from xml.sax.saxutils import escape
from datetime import datetime

# PDF (reportlab)
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    ListFlowable, ListItem
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Gemini (new SDK)
from google import genai

# .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# CONFIG
st.set_page_config(page_title="Xiaomi SA Dashboard", layout="wide")

PROJECT_ROOT = Path(".").resolve()
DATA_PATH = PROJECT_ROOT / "Output Analysis" / "Sentiment Analysis Dataset" / "Sentiment_Analyzed_setfit_final.csv"
ART_DIR = PROJECT_ROOT / "artifacts" / "summary"
ART_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL_FINAL = "sentiment_label_final"
VALID_LABELS = ["negative", "neutral", "positive"]

# Fixed per note
MODEL_NAME = "gemini-2.5-pro"
TARGET_PRODUCT = "Xiaomi Note 15 Pro"

PDF_PATH = ART_DIR / "Xiaomi_Note_15_Post_Launch_Report.pdf"
TXT_PATH = ART_DIR / "Final_Recommendation.txt"


# ASPECT DICT
ASPECTS_DICT = {
    "kamera": [
        "kamera", "cam", "leica", "kamera leica", "boba",
        "foto", "video", "rekam",
        "selfie", "kamera depan", "kamera belakang",
        "night", "night mode", "low light", "malam",
        "zoom", "tele", "telephoto", "ultrawide",
        "potret", "portrait", "bokeh",
        "ois", "eis",
        "4k", "8k",
        "200mp", "108mp", "50mp"
    ],
    "baterai": [
        "baterai", "battery", "batre", "batrei",
        "daya tahan", "kapasitas",
        "5000mah", "5500mah", "6000mah",
        "screen on time", "sot",
        "tahan seharian", "dipakai seharian"
    ],
    "charging": [
        "charge", "charging", "cas", "ngecas",
        "fast charging", "pengisian cepat",
        "hypercharge", "turbo charge",
        "67w", "90w", "120w",
        "wireless charging", "reverse charging"
    ],
    "thermal": [
        "panas", "overheat", "hangat", "gerah",
        "thermal", "throttling",
        "panas gaming", "panas main game",
        "panas saat cas",
        "cooling", "vapor chamber", "liquid cool"
    ],
    "performa": [
        "performa", "kinerja",
        "chipset", "soc", "processor",
        "snapdragon", "dimensity",
        "antutu", "geekbench",
        "gaming", "fps", "frame drop",
        "multitasking",
        "pubg", "genshin", "mobile legends", "ml", "cod"
    ],
    "software": [
        "hyperos", "miui",
        "software", "sistem",
        "ui", "antarmuka",
        "update", "pembaruan",
        "bug", "error",
        "bloatware", "iklan",
        "fitur", "gesture", "animasi"
    ],
    "layar": [
        "layar", "screen",
        "amoled", "oled", "ltpo",
        "refresh rate", "hz", "120hz", "144hz",
        "resolusi", "1.5k", "2k",
        "brightness", "nits",
        "hdr", "dolby vision",
        "fingerprint", "sidik jari",
        "pwm", "eye care"
    ],
    "desain": [
        "desain", "design",
        "build", "build quality",
        "bodi", "body", "frame",
        "material",
        "kaca", "glass",
        "flat", "curve", "curved",
        "bezel", "tebal", "tipis",
        "warna", "color", "finishing"
    ],
    "harga": [
        "harga", "price",
        "jutaan", "rp",
        "murah", "mahal",
        "worth", "worth it",
        "value", "value for money",
        "promo", "diskon", "flash sale",
        "shopee", "tokopedia", "lazada"
    ],
}

# UTIL
def compile_aspect_patterns(aspects_dict):
    aspect_patterns = {}
    for aspect, keywords in aspects_dict.items():
        patterns = []
        seen = set()
        for kw in keywords:
            kw = kw.strip().lower()
            if not kw or kw in seen:
                continue
            seen.add(kw)
            if len(kw) <= 2:
                patterns.append(re.compile(rf"\b{re.escape(kw)}\b"))
            else:
                patterns.append(re.compile(rf"(?<!\w){re.escape(kw)}(?!\w)"))
        aspect_patterns[aspect] = patterns
    return aspect_patterns


ASPECT_PATTERNS = compile_aspect_patterns(ASPECTS_DICT)


def tag_aspects(text: str):
    if not isinstance(text, str) or not text.strip():
        return ["umum"]
    t = text.lower()
    tags = []
    for aspect, patterns in ASPECT_PATTERNS.items():
        if any(p.search(t) for p in patterns):
            tags.append(aspect)
    return tags if tags else ["umum"]


def md_to_rl_safe(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!<)\*(.+?)\*(?!>)", r"<i>\1</i>", text)
    return text


@st.cache_data(show_spinner=False)
def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"CSV tidak ditemukan: {path}")
    df = pd.read_csv(path)

    if "clean_text" not in df.columns:
        raise ValueError("Kolom `clean_text` tidak ada.")
    if LABEL_COL_FINAL not in df.columns:
        raise ValueError(f"Kolom `{LABEL_COL_FINAL}` tidak ada di dataset.")

    df[LABEL_COL_FINAL] = df[LABEL_COL_FINAL].astype(str).str.lower().str.strip()
    df = df[df[LABEL_COL_FINAL].isin(VALID_LABELS)].copy()

    label_to_polarity = {"negative": -1, "neutral": 0, "positive": 1}
    df["polarity"] = df[LABEL_COL_FINAL].map(label_to_polarity).astype(int)

    df["aspects"] = df["clean_text"].astype(str).apply(tag_aspects)
    df_exploded = df.explode("aspects").copy()
    return df, df_exploded


def make_aspect_summary(df_exploded):
    aspect_summary = (
        df_exploded
        .groupby("aspects", dropna=False)
        .agg(
            total_mentions=("clean_text", "count"),
            avg_sentiment=("polarity", "mean"),
        )
        .sort_values("total_mentions", ascending=False)
    )
    mean = aspect_summary["avg_sentiment"].mean()
    std = aspect_summary["avg_sentiment"].std()
    if std == 0 or np.isnan(std):
        aspect_summary["z_score"] = 0.0
    else:
        aspect_summary["z_score"] = (aspect_summary["avg_sentiment"] - mean) / std
    return aspect_summary


def make_aspect_distribution(df_exploded):
    dist = (
        df_exploded
        .groupby(["aspects", LABEL_COL_FINAL])
        .size()
        .unstack(fill_value=0)
    )
    for col in ["negative", "neutral", "positive"]:
        if col not in dist.columns:
            dist[col] = 0
    dist["total_mentions"] = dist[["negative", "neutral", "positive"]].sum(axis=1)
    dist["negative_rate"] = dist["negative"] / dist["total_mentions"]
    dist["neutral_rate"]  = dist["neutral"]  / dist["total_mentions"]
    dist["positive_rate"] = dist["positive"] / dist["total_mentions"]
    return dist


def fig_aspect_zscore_bar(aspect_summary):
    asp = aspect_summary.sort_values("z_score", ascending=False)
    fig = plt.figure(figsize=(10, 4))
    plt.bar(asp.index, asp["z_score"])
    plt.axhline(0)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Z-Score")
    plt.title("Aspect-Level Sentiment Distribution (Z-Score Based)")
    plt.tight_layout()
    return fig


def fig_top_pain_positive(aspect_distribution, min_mentions=30, topk=5):
    base = aspect_distribution.query("total_mentions >= @min_mentions").copy()

    top_pain = (
        base.sort_values(["negative_rate", "total_mentions"], ascending=False)
            .head(topk)
            .reset_index()
    )
    top_pos = (
        base.sort_values(["positive_rate", "total_mentions"], ascending=False)
            .head(topk)
            .reset_index()
    )

    fig1 = plt.figure(figsize=(8, 3.8))
    sns.barplot(data=top_pain, x="negative_rate", y="aspects", hue="aspects", palette="Reds_r", legend=False)
    plt.xlim(0, 1)
    plt.xlabel("Proporsi Komentar Negatif")
    plt.ylabel("")
    plt.title(f"Top {topk} Pain Points (min mentions={min_mentions})")
    plt.tight_layout()

    fig2 = plt.figure(figsize=(8, 3.8))
    sns.barplot(data=top_pos, x="positive_rate", y="aspects", hue="aspects", palette="Greens", legend=False)
    plt.xlim(0, 1)
    plt.xlabel("Proporsi Komentar Positif")
    plt.ylabel("")
    plt.title(f"Top {topk} Positive Points (min mentions={min_mentions})")
    plt.tight_layout()

    return fig1, fig2, top_pain, top_pos


def fig_prioritization_matrix(df_exploded, min_mentions=30):
    aspect_metrics = (
        df_exploded
        .groupby("aspects")
        .agg(
            frequency=("clean_text", "count"),
            avg_sentiment=("polarity", "mean")
        )
        .reset_index()
    )
    aspect_metrics = aspect_metrics[aspect_metrics["frequency"] >= min_mentions].copy()

    if aspect_metrics["avg_sentiment"].std() == 0:
        aspect_metrics["sentiment_z"] = 0
    else:
        aspect_metrics["sentiment_z"] = zscore(aspect_metrics["avg_sentiment"])

    fig = plt.figure(figsize=(12, 7))
    sns.scatterplot(
        data=aspect_metrics,
        x="sentiment_z",
        y="frequency",
        size="frequency",
        sizes=(200, 1200),
        hue="sentiment_z",
        palette="RdYlGn",
        edgecolor="black",
        alpha=0.85,
        legend=False
    )

    for _, row in aspect_metrics.iterrows():
        plt.text(row["sentiment_z"] + 0.05, row["frequency"], row["aspects"], fontsize=10, weight="bold")

    plt.axvline(0, color="black", linestyle="--", alpha=0.6)
    freq_median = aspect_metrics["frequency"].median()
    plt.axhline(freq_median, color="black", linestyle="--", alpha=0.6)

    y_max = aspect_metrics["frequency"].max()
    plt.text(-2, y_max, "🚨 CRITICAL ISSUES\nHigh Volume • Negative", color="red", fontsize=10,
             bbox=dict(facecolor="white", alpha=0.8))
    plt.text(1.2, y_max, "💎 SELLING POINTS\nHigh Volume • Positive", color="green", fontsize=10,
             bbox=dict(facecolor="white", alpha=0.8))
    plt.text(-2, freq_median/2, "⚠ MINOR ISSUES\nLow Volume • Negative", color="darkred", fontsize=9,
             bbox=dict(facecolor="white", alpha=0.7))
    plt.text(1.2, freq_median/2, "🌱 NICE TO HAVE\nLow Volume • Positive", color="darkgreen", fontsize=9,
             bbox=dict(facecolor="white", alpha=0.7))

    plt.title("Prioritization Matrix: Impact vs Sentiment (Final SetFit)", fontsize=14)
    plt.xlabel("Sentiment Score (Z-score)")
    plt.ylabel("Frequency (Volume Pembicaraan)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def fig_cross_channel_heatmap(df_exploded, min_mentions=30):
    if "source_channel" not in df_exploded.columns:
        raise ValueError("Kolom `source_channel` tidak ada di dataset.")

    aspect_counts = df_exploded.groupby("aspects").size()
    valid_aspects = aspect_counts[aspect_counts >= min_mentions].index
    df_heat = df_exploded[df_exploded["aspects"].isin(valid_aspects)].copy()

    heatmap_data = df_heat.pivot_table(
        index="aspects",
        columns="source_channel",
        values="polarity",
        aggfunc="mean"
    )

    fig = plt.figure(figsize=(12, 7))
    mask = heatmap_data.isna()
    sns.heatmap(
        heatmap_data,
        mask=mask,
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Avg Sentiment Score (-1 → +1)"}
    )
    plt.title("Cross-Channel Sentiment Heatmap\n(Konsistensi Persepsi Antar Reviewer)", fontsize=13)
    plt.ylabel("Aspek Produk")
    plt.xlabel("Channel YouTube")
    plt.tight_layout()
    return fig


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def top_keywords_per_aspect(df_exploded, aspect_list, topn=5):
    """
    Minimal & explainable:
    - ambil clean_text per aspek
    - tokenisasi whitespace
    - ambil top kata (panjang >= 3) untuk mengurangi noise
    """
    out = {}
    for asp in aspect_list:
        sub = df_exploded[df_exploded["aspects"] == asp]["clean_text"].astype(str)
        tokens = []
        for t in sub.tolist():
            tokens.extend([w for w in t.lower().split() if len(w) >= 3])
        cnt = Counter(tokens)
        out[asp] = [w for w, _ in cnt.most_common(topn)]
    return out


def generate_report_text(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY belum diset. Isi di .env atau env var.")

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    text = getattr(resp, "text", None) or str(resp)

    TXT_PATH.write_text(text, encoding="utf-8")
    return text


def build_pdf_from_text(raw_text: str, output_pdf: Path):
    lines = [line.rstrip() for line in (raw_text or "").split("\n")]

    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=2.8*cm,
        rightMargin=2.8*cm,
        topMargin=3.5*cm,
        bottomMargin=2.5*cm,
    )

    styles = getSampleStyleSheet()

    def add_style(name, **kwargs):
        if name not in styles:
            styles.add(ParagraphStyle(name=name, **kwargs))

    add_style("CoverMainTitle", fontSize=26, leading=30, alignment=TA_CENTER, spaceAfter=12,
              textColor=HexColor("#0D47A1"), fontName="Helvetica-Bold")
    add_style("CoverProduct", fontSize=20, leading=24, alignment=TA_CENTER, spaceAfter=0,
              textColor=HexColor("#1565C0"), fontName="Helvetica")

    add_style("Section", fontSize=15.5, leading=19, spaceBefore=16, spaceAfter=7,
              textColor=HexColor("#1565C0"), fontName="Helvetica-Bold")
    add_style("Subsection", fontSize=13, leading=16, spaceBefore=12, spaceAfter=6, leftIndent=8,
              textColor=HexColor("#424242"), fontName="Helvetica-Bold")
    add_style("Body", fontSize=10.7, leading=14.5, alignment=TA_JUSTIFY, spaceAfter=7, spaceBefore=3)
    add_style("Bullet", fontSize=10.5, leading=14, leftIndent=26, bulletIndent=13, spaceAfter=5)

    story = []
    story.append(Spacer(1, 8*cm))
    story.append(Paragraph("Post-Launch Market Intelligence Report", styles["CoverMainTitle"]))
    story.append(Spacer(1, 1.2*cm))
    story.append(Paragraph(TARGET_PRODUCT, styles["CoverProduct"]))
    story.append(Spacer(1, 10*cm))
    story.append(PageBreak())

    bullet_buffer = []
    prev_was_section = False

    def flush_bullets():
        if not bullet_buffer:
            return
        items = [ListItem(Paragraph(md_to_rl_safe(txt.strip()), styles["Bullet"])) for txt in bullet_buffer]
        story.append(ListFlowable(
            items,
            bulletType="bullet",
            bulletFontName="Helvetica",
            bulletFontSize=10.8,
            leftPadding=6,
            spaceBetween=3.5
        ))
        bullet_buffer.clear()

    for line in lines:
        stripped = line.strip()

        if re.match(r"^[-—━•*_=]{3,}$", stripped) or not stripped:
            flush_bullets()
            story.append(Spacer(1, 5))
            continue

        if stripped.startswith("### "):
            flush_bullets()
            if prev_was_section:
                story.append(PageBreak())
            else:
                story.append(Spacer(1, 8))
            section_title = stripped[4:].strip()
            story.append(Paragraph(md_to_rl_safe(section_title), styles["Section"]))
            story.append(Spacer(1, 6))
            prev_was_section = True
            continue

        if (stripped.startswith("**") and stripped.endswith("**")) or \
           (stripped.endswith(":") and len(stripped) < 70 and not stripped.isupper()):
            flush_bullets()
            clean = stripped.strip(":").strip()
            story.append(Paragraph(md_to_rl_safe(clean), styles["Subsection"]))
            story.append(Spacer(1, 5))
            continue

        if stripped.startswith(("-", "•", "*", "→", "–")) or re.match(r"^\s*[-*•→–]\s", line):
            bullet_text = re.sub(r"^[-*•→–]\s*", "", stripped).strip()
            bullet_buffer.append(bullet_text)
            continue

        flush_bullets()
        story.append(Paragraph(md_to_rl_safe(stripped), styles["Body"]))
        story.append(Spacer(1, 4))

    flush_bullets()
    doc.build(story)
    return output_pdf

# UI
st.title("Xiaomi Sentiment Analysis — Dashboard")

try:
    df, df_exploded = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Gagal load data: {e}")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.header("Controls")

    # 1) tetap sesuai: MIN_MENTIONS & Top-K
    min_mentions = st.slider("MIN_MENTIONS (filter noise aspek)", 5, 100, 30, 5)
    topk = st.slider("Top-K pain/positive", 3, 15, 5, 1)

    st.markdown("---")

    # 2) urutan sesuai: Search → Channel → Sentiment → Aspect, dibuat rapi via expander
    with st.expander("🔎 Filters", expanded=True):
        keyword = st.text_input("Search keyword (optional)", value="")

        if "source_channel" in df.columns:
            channels = sorted(df["source_channel"].dropna().unique().tolist())
        else:
            channels = []
        selected_channels = st.multiselect("Filter Channel", options=channels, default=channels)

        selected_sentiments = st.multiselect("Filter Sentiment", options=VALID_LABELS, default=VALID_LABELS)

        all_aspects = sorted(set(df_exploded["aspects"].dropna().astype(str).tolist()))
        selected_aspects = st.multiselect("Filter Aspect", options=all_aspects, default=all_aspects)

# Apply filters
df_view = df.copy()

if selected_channels and "source_channel" in df_view.columns:
    df_view = df_view[df_view["source_channel"].isin(selected_channels)]

df_view = df_view[df_view[LABEL_COL_FINAL].isin(selected_sentiments)]

if keyword.strip():
    kw = keyword.strip().lower()
    if "text" in df_view.columns:
        mask = df_view["text"].astype(str).str.lower().str.contains(kw, na=False)
    else:
        mask = df_view["clean_text"].astype(str).str.lower().str.contains(kw, na=False)
    mask = mask | df_view["clean_text"].astype(str).str.lower().str.contains(kw, na=False)
    df_view = df_view[mask]

# rebuild exploded agar chart ikut filter + aspect filter
df_view["aspects"] = df_view["clean_text"].astype(str).apply(tag_aspects)
df_view_exploded = df_view.explode("aspects").copy()
if selected_aspects:
    df_view_exploded = df_view_exploded[df_view_exploded["aspects"].isin(selected_aspects)].copy()

# KPI Cards
total_com = int(len(df_view))
sent_counts = df_view[LABEL_COL_FINAL].value_counts()
pos_n = int(sent_counts.get("positive", 0))
neg_n = int(sent_counts.get("negative", 0))
neu_n = int(sent_counts.get("neutral", 0))

sent_pct = df_view[LABEL_COL_FINAL].value_counts(normalize=True) * 100
pos_pct = float(sent_pct.get("positive", 0.0))
neg_pct = float(sent_pct.get("negative", 0.0))
neu_pct = float(sent_pct.get("neutral", 0.0))

net_sentiment = pos_pct - neg_pct

# Top keywords per aspect (memakai aspek teratas by mention setelah filter)
aspect_summary_tmp = make_aspect_summary(df_view_exploded)
top_aspects_for_kw = aspect_summary_tmp.head(3).index.tolist()
kw_map = top_keywords_per_aspect(df_view_exploded, top_aspects_for_kw, topn=5)

def fmt_int(n): return f"{n:,}"
def fmt_pct(p): return f"{p:.1f}%"

# simple CSS card (stable)
st.markdown(
    """
    <style>
      .kpi-grid {display:grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-top: 8px; margin-bottom: 8px;}
      .kpi-card {border: 1px solid rgba(0,0,0,.08); border-radius: 14px; padding: 12px 14px; background: rgba(255,255,255,.02);}
      .kpi-title {font-size: 12px; opacity: .75; margin-bottom: 6px;}
      .kpi-main {font-size: 22px; font-weight: 700; line-height: 1.2;}
      .kpi-sub {font-size: 12px; opacity: .8; margin-top: 4px;}
      .kpi-pill {display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; background: rgba(59,130,246,.10);}
    </style>
    """,
    unsafe_allow_html=True
)

kpi_html = f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-title">Total Comments</div>
    <div class="kpi-main">{fmt_int(total_com)}</div>
    <div class="kpi-sub">Filtered view</div>
  </div>

  <div class="kpi-card">
    <div class="kpi-title">Positive</div>
    <div class="kpi-main">{fmt_int(pos_n)}</div>
    <div class="kpi-sub">{fmt_pct(pos_pct)}</div>
  </div>

  <div class="kpi-card">
    <div class="kpi-title">Negative</div>
    <div class="kpi-main">{fmt_int(neg_n)}</div>
    <div class="kpi-sub">{fmt_pct(neg_pct)}</div>
  </div>

  <div class="kpi-card">
    <div class="kpi-title">Neutral</div>
    <div class="kpi-main">{fmt_int(neu_n)}</div>
    <div class="kpi-sub">{fmt_pct(neu_pct)}</div>
  </div>

  <div class="kpi-card">
    <div class="kpi-title">Net Sentiment</div>
    <div class="kpi-main">{net_sentiment:.1f} <span class="kpi-pill">/100</span></div>
    <div class="kpi-sub">Pos% − Neg%</div>
  </div>

  <div class="kpi-card">
    <div class="kpi-title">Top Keywords (Top Aspects)</div>
    <div class="kpi-sub">
      {"<br>".join([f"<b>{a}</b>: {', '.join(kw_map.get(a, []))}" for a in top_aspects_for_kw])}
    </div>
  </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

st.markdown("---")


# 1) TABLE + Export CSV
st.subheader("1) Dataset Table (Final SetFit)")

table_cols = []
for c in ["source_channel", "author", "text", LABEL_COL_FINAL]:
    if c in df_view.columns:
        table_cols.append(c)

if not table_cols:
    st.warning("Kolom table minimum tidak ditemukan (source_channel/author/text/sentiment_label_final).")
else:
    rename_map = {
        "source_channel": "Channel YouTube",
        "author": "Nama User",
        "text": "Komentar Asli",
        LABEL_COL_FINAL: "Sentiment Final",
    }
    df_table = df_view[table_cols].rename(columns=rename_map)

    st.dataframe(df_table, use_container_width=True, height=380)

    csv_bytes = df_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Export CSV (filtered table)",
        data=csv_bytes,
        file_name="filtered_table.csv",
        mime="text/csv"
    )

# 2) Aspect Z-score + download
st.subheader("2) Aspect-Level Sentiment Distribution (Z-Score Based)")
aspect_summary = make_aspect_summary(df_view_exploded)
fig2 = fig_aspect_zscore_bar(aspect_summary)
st.pyplot(fig2)

st.download_button(
    "⬇️ Download chart PNG",
    data=fig_to_png_bytes(fig2),
    file_name="chart_aspect_zscore.png",
    mime="image/png"
)

# 3) Pain & Positive Points + download
st.subheader("3) Top Pain Points & Positive Points")
aspect_dist = make_aspect_distribution(df_view_exploded)
fig3a, fig3b, _, _ = fig_top_pain_positive(aspect_dist, min_mentions=min_mentions, topk=topk)

c1, c2 = st.columns(2)
with c1:
    st.pyplot(fig3a)
    st.download_button(
        "⬇️ PNG",
        data=fig_to_png_bytes(fig3a),
        file_name="chart_top_pain_points.png",
        mime="image/png"
    )
with c2:
    st.pyplot(fig3b)
    st.download_button(
        "⬇️ PNG",
        data=fig_to_png_bytes(fig3b),
        file_name="chart_top_positive_points.png",
        mime="image/png"
    )

# 4) Prioritization Matrix + download
st.subheader("4) Prioritization Matrix")
fig4 = fig_prioritization_matrix(df_view_exploded, min_mentions=min_mentions)
st.pyplot(fig4)

st.download_button(
    "⬇️ Download chart PNG",
    data=fig_to_png_bytes(fig4),
    file_name="chart_prioritization_matrix.png",
    mime="image/png"
)

# 5) Heatmap + download
st.subheader("5) Cross-Channel Sentiment Heatmap (Consistency)")
if "source_channel" in df_view_exploded.columns and df_view_exploded["source_channel"].notna().any():
    fig5 = fig_cross_channel_heatmap(df_view_exploded, min_mentions=min_mentions)
    st.pyplot(fig5)

    st.download_button(
        "⬇️ Download chart PNG",
        data=fig_to_png_bytes(fig5),
        file_name="chart_heatmap_consistency.png",
        mime="image/png"
    )
else:
    st.info("Heatmap butuh kolom `source_channel`.")

# 6) Gemini → PDF + Download
st.subheader("6) Executive Report (Gemini) + Download PDF")

try:
    focus = aspect_summary[aspect_summary["total_mentions"] >= min_mentions]
    complaints = focus.sort_values("avg_sentiment", ascending=True).head(3)
    praises = focus.sort_values("avg_sentiment", ascending=False).head(3)
    top_complaints = ", ".join([f"{idx} ({row.avg_sentiment:.2f})" for idx, row in complaints.iterrows()])
    top_praises = ", ".join([f"{idx} ({row.avg_sentiment:.2f})" for idx, row in praises.iterrows()])
except Exception:
    top_complaints = "Belum tersedia"
    top_praises = "Belum tersedia"

today_str = datetime.now().strftime("%d %B %Y")

prompt = f"""
Hari ini: {today_str}

Anda adalah Senior Data Scientist di Xiaomi Global Market Intelligence Team.
Bayangkan Anda sedang menyusun laporan untuk meeting internal bersama:
Product Manager, Head of Marketing, dan Lead MIUI Engineer.

Tugas Anda adalah menyusun:
POST-LAUNCH MARKET INTELLIGENCE REPORT
untuk produk: {TARGET_PRODUCT}

Gunakan bahasa yang profesional namun tetap natural, tidak terlalu kaku,
mudah dipahami oleh tim lintas fungsi (non-technical & technical).

====================================================
RINGKASAN DATA (YouTube Sentiment Intelligence)
====================================================

Total Komentar Dianalisis: {total_com}
Brand Health Index (Net Sentiment): {net_sentiment:.1f} / 100

Distribusi Sentimen:
• Positif : {pos_pct:.1f}%
• Negatif : {neg_pct:.1f}%
• Netral  : {neu_pct:.1f}%

Winning Features (Paling Banyak Dipuji):
{top_praises}

Critical Pain Points (Paling Banyak Dikeluhkan):
{top_complaints}

====================================================
INSTRUKSI PENYUSUNAN LAPORAN
====================================================

Buat laporan dengan struktur berikut:

1. EXECUTIVE SUMMARY
   - Tentukan status peluncuran: SUCCESS / WARNING / CRITICAL
   - Jelaskan secara ringkas bagaimana penerimaan pasar saat ini
   - Sorot 3 insight paling penting (bullet points)

2. MARKET SENTIMENT ANALYSIS
   - Apa yang paling disukai user? Kenapa itu penting secara bisnis?
   - Apa yang paling mengganggu user? Apa potensi risikonya?

3. PRODUCT & HARDWARE ACTION PLAN
   - Rekomendasi konkret dan realistis
   - Urutkan berdasarkan prioritas (High / Medium / Low Impact)
   - Jelaskan trade-off jika ada

4. SOFTWARE (MIUI / SYSTEM) ACTION PLAN
   - Apakah ada indikasi bug, optimasi, atau UX friction?
   - Rekomendasi update OTA atau patch strategy

5. MARKETING & PR STRATEGY
   - Fitur apa yang harus lebih di-highlight?
   - Narasi apa yang perlu diperkuat?
   - Apakah perlu counter-narrative terhadap keluhan tertentu?

6. STRATEGIC OUTLOOK (Forward Looking)
   - Jika tren ini berlanjut 3–6 bulan ke depan, apa implikasinya?
   - Apakah positioning produk ini sustainable?

====================================================
GAYA PENULISAN
====================================================

- Gunakan tone profesional namun conversational
- Hindari kalimat generik
- Fokus pada insight, bukan hanya rekap data
- Berikan rekomendasi yang actionable dan realistis
- Jangan bertele-tele, tapi tetap mendalam

Bayangkan laporan ini akan dibaca langsung oleh Country Manager.

Tuliskan laporan secara lengkap dan terstruktur rapi.
"""

if "report_text" not in st.session_state:
    st.session_state["report_text"] = ""
if "pdf_ready" not in st.session_state:
    st.session_state["pdf_ready"] = False

if st.button("⚡ Generate Report (Gemini) → PDF"):
    try:
        with st.spinner(f"Generate report menggunakan {MODEL_NAME}..."):
            text = generate_report_text(prompt)
            st.session_state["report_text"] = text

        with st.spinner("Build PDF..."):
            build_pdf_from_text(st.session_state["report_text"], PDF_PATH)
            st.session_state["pdf_ready"] = True

        st.success("Report berhasil dibuat & PDF siap di-download.")
    except Exception as e:
        st.error(f"Gagal generate report: {e}")

# Preview proper
if st.session_state["report_text"].strip():
    st.markdown("### Preview Report")
    st.text_area("Report", value=st.session_state["report_text"], height=320)

# Download PDF
if PDF_PATH.exists() and st.session_state["pdf_ready"]:
    st.download_button(
        "⬇️ Download PDF Report",
        data=PDF_PATH.read_bytes(),
        file_name=PDF_PATH.name,
        mime="application/pdf"
    )
else:
    st.caption("PDF belum tersedia. Klik tombol Generate Report (Gemini) → PDF.")