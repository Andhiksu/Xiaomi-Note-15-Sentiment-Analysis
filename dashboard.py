import os
import re
import io
import json
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

# Gemini
from google import genai

# .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# Konfigurasi Streamlit
st.set_page_config(page_title="Xiaomi Sentiment Dashboard", page_icon="📊", layout="wide")

# Set seaborn theme
sns.set_theme(style="whitegrid", palette="muted")

PROJECT_ROOT = Path(".").resolve()
DATA_PATH = PROJECT_ROOT / "dungeon" / "processed_dataset" / "sentiment_analyzed_setfit_final.csv"

ART_DIR = PROJECT_ROOT / "dungeon" / "summary_analysis" / "dashboard_outputs"
ART_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL_FINAL = "sentiment_label_final"
VALID_LABELS = ["negative", "neutral", "positive"]

MODEL_NAME = "gemini-3.1-pro-preview"
TARGET_PRODUCT = "Xiaomi Note 15 5G"

PDF_PATH = ART_DIR / "Xiaomi_Note_15_Post_Launch_Report.pdf"
TXT_PATH = ART_DIR / "Final_Recommendation.txt"
PROMPT_PATH = ART_DIR / "Prompt_Used.txt"


# Aspek dan keyword terkait untuk tagging
ASPECTS_DICT = {
    "kamera": [
        "kamera", "cam", "leica", "boba", "lensa", "sensor",
        "foto", "video", "rekam", "cinematic", "vlog",
        "selfie", "kamera depan", "kamera belakang",
        "night", "malam", "low light", "gelap", "lampu",
        "zoom", "tele", "telephoto", "periskop", "ultrawide", "macro",
        "potret", "portrait", "bokeh", "blur", "ois", "eis", "stabilizer",
        "4k", "8k", "60fps", "hdr", "shutter", "skin tone",
        "200mp", "50mp", "108mp", "imaging engine"
    ],
    "performa": [
        "performa", "kinerja", "ngebut", "lancar", "lag", "patah", "patah-patah",
        "chipset", "soc", "processor", "prosesor", "otak",
        "snapdragon", "dimensity", "gen 3", "gen 4", "8s", "mediatek",
        "antutu", "geekbench", "skor", "benchmark",
        "gaming", "game", "fps", "frame drop", "rendering", "rata kanan",
        "multitasking", "render", "ngefreeze",
        "pubg", "genshin", "mlbb", "mobile legends", "ff", "free fire", 
        "hok", "honor of kings", "wuwa", "wuthering waves", "codm"
    ],
    "layar": [
        "layar", "screen", "display", "panel", "bezel",
        "amoled", "oled", "ltpo", "c8", "c9",
        "refresh rate", "hz", "120hz", "144hz", "adaptif",
        "resolusi", "1.5k", "2k", "fhd", "tajam", "bening",
        "brightness", "nits", "terang", "silau", "outdoor",
        "hdr10", "dolby vision", "vivid",
        "fingerprint", "sidik jari", "under display",
        "pwm dimming", "eye care", "mata lelah", "flicker"
    ],
    "baterai": [
        "baterai", "battery", "batre", "batrei", "power",
        "daya tahan", "kapasitas", "awet", "boros", "drain", 
        "5000mah", "5100mah", "5500mah", "6000mah",
        "screen on time", "sot", "seharian", "standby"
    ],
    "charging": [
        "charge", "charging", "cas", "ngecas", "isi daya",
        "fast charging", "pengisian cepat", "hypercharge", "turbo charge",
        "67w", "90w", "120w", "kepala charger", "adaptor",
        "wireless charging", "nirkabel", "reverse charging"
    ],
    "software": [
        "hyperos", "miui", "android", "os", "sistem",
        "software", "ui", "antarmuka", "update", "pembaruan",
        "bug", "error", "bloatware", "iklan", "aplikasi bawaan",
        "fitur", "gesture", "animasi", "control center",
        "ai", "artificial intelligence", "ai eraser", "ai portrait", 
        "ai expansion", "smart", "pintar", "circle to search"
    ],
    "desain": [
        "desain", "design", "tampang", "bentuk", "model",
        "build", "build quality", "kokoh", "mewah", "premium",
        "bodi", "body", "frame", "material", "tekstur",
        "kaca", "glass", "vegan leather", "kulit", "titanium",
        "flat", "curve", "curved", "melengkung", "kotak",
        "tipis", "ringan", "berat", "tebal",
        "warna", "color", "finishing", "matte", "glossy",
        "ip68", "ip69", "tahan air", "debu", "gorilla glass"
    ],
    "thermal": [
        "panas", "overheat", "hangat", "adem", "dingin", "gerah",
        "thermal", "throttling", "suhu", "celcius",
        "cooling", "vapor chamber", "liquid cool", "iceloop",
        "panas gaming", "panas cas"
    ],
    "harga": [
        "harga", "price", "rp", "jutaan", "pasaran",
        "murah", "mahal", "worth", "worth it", "layak",
        "value", "value for money", "price to performance",
        "promo", "diskon", "flash sale", "pre order", "po",
        "mending", "saingan", "lawan", "kompetitor",
        "shopee", "tokopedia", "lazada", "blibli", "tiktok shop", "toko"
    ],
}

# Util Functions
@st.cache_resource
def compile_aspect_patterns(aspects_dict):
    aspect_patterns = {}
    for aspect, keywords in aspects_dict.items():
        patterns = []
        seen = set()
        for kw in keywords:
            kw = kw.strip().lower()
            if not kw or kw in seen: continue
            seen.add(kw)
            if len(kw) <= 2:
                patterns.append(re.compile(rf"\b{re.escape(kw)}\b"))
            else:
                patterns.append(re.compile(rf"(?<!\w){re.escape(kw)}(?!\w)"))
        aspect_patterns[aspect] = patterns
    return aspect_patterns

ASPECT_PATTERNS = compile_aspect_patterns(ASPECTS_DICT)

def tag_aspects(text: str):
    if not isinstance(text, str) or not text.strip(): return ["umum"]
    t = text.lower()
    tags = [aspect for aspect, patterns in ASPECT_PATTERNS.items() if any(p.search(t) for p in patterns)]
    return tags if tags else ["umum"]

def md_to_rl_safe(text: str) -> str:
    if not isinstance(text, str): return ""
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = text.replace("*", "")
    return text

@st.cache_data(show_spinner=False)
def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"CSV tidak ditemukan: {path}")

    df = pd.read_csv(path)
    if "clean_text" not in df.columns: raise ValueError("Kolom `clean_text` tidak ada.")
    if LABEL_COL_FINAL not in df.columns: raise ValueError(f"Kolom `{LABEL_COL_FINAL}` tidak ada.")

    df["clean_text"] = df["clean_text"].astype(str).str.strip()
    df[LABEL_COL_FINAL] = df[LABEL_COL_FINAL].astype(str).str.lower().str.strip()
    df = df[(df["clean_text"] != "") & (df[LABEL_COL_FINAL].isin(VALID_LABELS))].copy()

    label_to_polarity = {"negative": -1, "neutral": 0, "positive": 1}
    df["polarity"] = df[LABEL_COL_FINAL].map(label_to_polarity)
    df["aspects"] = df["clean_text"].apply(tag_aspects)

    df_exploded = df.explode("aspects").copy()
    df_exploded["aspects"] = df_exploded["aspects"].astype(str).str.strip().str.lower()
    return df, df_exploded
# Aspek Summary dengan Z-Score untuk Bar Chart
def make_aspect_summary(df_exploded):
    df_use = df_exploded[df_exploded["aspects"] != "umum"].copy()
    aspect_summary = (
        df_use.groupby("aspects")
        .agg(total_mentions=("clean_text", "count"), avg_sentiment=("polarity", "mean"))
        .sort_values("total_mentions", ascending=False)
    )
    if aspect_summary.empty: return aspect_summary

    std_val = aspect_summary["avg_sentiment"].std()
    aspect_summary["z_score"] = 0 if std_val == 0 or np.isnan(std_val) else zscore(aspect_summary["avg_sentiment"])
    return aspect_summary
# Aspek Distribution untuk Bar Chart & Heatmap
def make_aspect_distribution(df_exploded):
    df_use = df_exploded[df_exploded["aspects"] != "umum"].copy()
    dist = df_use.groupby(["aspects", LABEL_COL_FINAL]).size().unstack(fill_value=0)
    
    for col in ["negative", "neutral", "positive"]:
        if col not in dist.columns: dist[col] = 0

    dist["total_mentions"] = dist[["negative","neutral","positive"]].sum(axis=1)
    dist["negative_rate"] = dist["negative"] / dist["total_mentions"]
    dist["positive_rate"] = dist["positive"] / dist["total_mentions"]
    dist["neutral_rate"] = dist["neutral"] / dist["total_mentions"]
    return dist.sort_values("total_mentions", ascending=False)

# Plotting Functions
def fig_aspect_zscore_bar(aspect_summary):
    asp = aspect_summary.sort_values("z_score", ascending=False)
    colors = ['#2ecc71' if z > 0 else '#e74c3c' for z in asp["z_score"]]
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(asp.index, asp["z_score"], color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=1.2)
    ax.set_xticklabels(asp.index, rotation=45, ha="right")
    ax.set_ylabel("Z-Score")
    ax.set_title("Aspect-Level Sentiment Distribution", pad=15, weight='bold')
    plt.tight_layout()
    return fig

def fig_top_pain_positive(aspect_distribution, min_mentions=30, topk=5):
    base = aspect_distribution.query("total_mentions >= @min_mentions").copy()
    top_pain = base.sort_values(["negative_rate", "total_mentions"], ascending=False).head(topk).reset_index()
    top_pos = base.sort_values(["positive_rate", "total_mentions"], ascending=False).head(topk).reset_index()

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=top_pain, x="negative_rate", y="aspects", hue="aspects", legend=False, palette="Reds_r", ax=ax1)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("Proporsi Komentar Negatif")
    ax1.set_ylabel("")
    ax1.set_title(f"Top {topk} Pain Points", weight='bold')
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=top_pos, x="positive_rate", y="aspects", hue="aspects", legend=False, palette="Greens_r", ax=ax2)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Proporsi Komentar Positif")
    ax2.set_ylabel("")
    ax2.set_title(f"Top {topk} Positive Points", weight='bold')
    plt.tight_layout()

    return fig1, fig2

def fig_prioritization_matrix(df_exploded, min_mentions=30):
    df_use = df_exploded[df_exploded["aspects"] != "umum"]
    aspect_metrics = df_use.groupby("aspects").agg(frequency=("clean_text","count"), avg_sentiment=("polarity","mean")).reset_index()
    aspect_metrics = aspect_metrics.query("frequency >= @min_mentions")

    if aspect_metrics.empty:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.text(0.5, 0.5, "No aspect meets threshold", ha="center")
        ax.axis("off")
        return fig

    aspect_metrics["sentiment_z"] = zscore(aspect_metrics["avg_sentiment"])
    aspect_metrics["color"] = aspect_metrics["sentiment_z"].apply(lambda z: "#e74c3c" if z < -0.5 else ("#f39c12" if z < 0.5 else "#2ecc71"))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(aspect_metrics["sentiment_z"], aspect_metrics["frequency"], s=aspect_metrics["frequency"]*3, 
               c=aspect_metrics["color"], edgecolor="white", alpha=0.85, linewidth=1)

    for _, row in aspect_metrics.iterrows():
        ax.text(row["sentiment_z"] + 0.05, row["frequency"], row["aspects"], fontsize=9, weight="bold", alpha=0.8)

    ax.axvline(0, linestyle="--", color='grey', alpha=0.5)
    ax.axhline(aspect_metrics["frequency"].median(), linestyle="--", color='grey', alpha=0.5)

    ax.set_title("Prioritization Matrix: Impact vs Frequency", weight='bold', pad=15)
    ax.set_xlabel("Sentiment Z-Score")
    ax.set_ylabel("Mentions Frequency")
    
    # Quadrant Labels
    ax.text(ax.get_xlim()[0] + 0.1, ax.get_ylim()[1] * 0.95, "Urgent Fix", color="#e74c3c", weight='bold', alpha=0.7)
    ax.text(ax.get_xlim()[1] - 0.5, ax.get_ylim()[1] * 0.95, "Key Strengths", color="#2ecc71", weight='bold', alpha=0.7)

    plt.tight_layout()
    return fig

def fig_cross_channel_heatmap(df_exploded, min_mentions=30):
    if "source_channel" not in df_exploded.columns: return None
    df_use = df_exploded[df_exploded["aspects"]!="umum"]
    aspect_counts = df_use.groupby("aspects").size()
    valid_aspects = aspect_counts[aspect_counts>=min_mentions].index
    df_heat = df_use[df_use["aspects"].isin(valid_aspects)]

    if df_heat.empty: return None
    heatmap_data = df_heat.pivot_table(index="aspects", columns="source_channel", values="polarity", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap="RdYlGn", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title("Cross-Channel Sentiment Consistency", weight='bold', pad=15)
    plt.tight_layout()
    return fig

def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

def top_keywords_per_aspect(df_exploded, aspect_list, topn=5):
    df_use = df_exploded[df_exploded["aspects"]!="umum"]
    out = {}
    for asp in aspect_list:
        sub = df_use[df_use["aspects"]==asp]["clean_text"]
        tokens = [w for t in sub.astype(str) for w in t.lower().split() if len(w)>=4]
        out[asp] = [w for w, _ in Counter(tokens).most_common(topn)]
    return out


# PDF & GEMINI Report Generation
def generate_report_text(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: raise ValueError("GEMINI_API_KEY belum diset.")
    client = genai.Client(api_key=api_key)
    
    # Generate Content Menggunakan Gemini
    resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    text = getattr(resp, "text", None) or str(resp)

    TXT_PATH.write_text(text, encoding="utf-8")
    PROMPT_PATH.write_text(prompt, encoding="utf-8")
    return text

def build_pdf_from_text(raw_text: str, output_pdf: Path):
    lines = [line.rstrip() for line in (raw_text or "").split("\n")]
    doc = SimpleDocTemplate(str(output_pdf), pagesize=A4, leftMargin=2.8*cm, rightMargin=2.8*cm, topMargin=3.5*cm, bottomMargin=2.5*cm)
    styles = getSampleStyleSheet()

    for name, kwargs in [
        ("CoverMainTitle", dict(fontSize=24, leading=28, alignment=TA_CENTER, spaceAfter=12, textColor=HexColor("#0D47A1"), fontName="Helvetica-Bold")),
        ("CoverProduct", dict(fontSize=18, leading=22, alignment=TA_CENTER, textColor=HexColor("#1565C0"), fontName="Helvetica")),
        ("Section", dict(fontSize=14, leading=18, spaceBefore=16, spaceAfter=8, textColor=HexColor("#1565C0"), fontName="Helvetica-Bold")),
        ("Subsection", dict(fontSize=12, leading=15, spaceBefore=10, spaceAfter=6, textColor=HexColor("#424242"), fontName="Helvetica-Bold")),
        ("Body", dict(fontSize=10.5, leading=14.5, alignment=TA_JUSTIFY, spaceAfter=7)),
        ("Bullet", dict(fontSize=10.5, leading=14, leftIndent=20, bulletIndent=10, spaceAfter=4))
    ]:
        if name not in styles: styles.add(ParagraphStyle(name=name, **kwargs))

    story = [Spacer(1, 8*cm), Paragraph("Post-Launch Market Intelligence Report", styles["CoverMainTitle"]), 
             Paragraph(TARGET_PRODUCT, styles["CoverProduct"]), Spacer(1, 10*cm), PageBreak()]

    bullet_buffer = []
    
    def flush_bullets():
        if bullet_buffer:
            items = [ListItem(Paragraph(md_to_rl_safe(txt), styles["Bullet"])) for txt in bullet_buffer]
            story.append(ListFlowable(items, bulletType="bullet", bulletFontSize=10.5, spaceBetween=3))
            bullet_buffer.clear()

    for line in lines:
        stripped = line.strip()
        if re.match(r"^[-—━•*_=]{3,}$", stripped) or not stripped:
            flush_bullets(); story.append(Spacer(1, 5)); continue

        if stripped.startswith("### ") or re.match(r"^\d+\.\s+[A-Z].*", stripped):
            flush_bullets()
            story.append(Spacer(1, 8))
            clean_title = stripped[4:].strip() if stripped.startswith("### ") else stripped
            story.append(Paragraph(md_to_rl_safe(clean_title), styles["Section"]))
            continue

        if stripped.startswith("**") and stripped.endswith("**"):
            flush_bullets()
            story.append(Paragraph(f"<b>{md_to_rl_safe(stripped.strip('*'))}</b>", styles["Subsection"]))
            continue

        if stripped.startswith(("-", "•", "*")) or re.match(r"^\s*[-*•]\s", line):
            bullet_buffer.append(re.sub(r"^[-*•]\s*", "", stripped).strip())
            continue

        flush_bullets()
        story.append(Paragraph(md_to_rl_safe(stripped), styles["Body"]))

    flush_bullets()
    doc.build(story)
    return output_pdf


# UI Rendering
st.title("📊 Xiaomi Sentiment Analysis Dashboard")

try:
    df, df_exploded = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Gagal load data: {e}\nPastikan file CSV tersedia di path yang benar.")
    st.stop()

# Sidebar Controls
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/ae/Xiaomi_logo_%282021-%29.svg", width=60)
    st.header("⚙️ Controls")

    min_mentions = st.slider("Min Mentions (Aspect Noise Filter)", 5, 100, 30, 5)
    topk = st.slider("Top-K Pain/Positive", 3, 10, 5, 1)

    st.markdown("---")
    with st.expander("🔎 Data Filters", expanded=True):
        keyword = st.text_input("Search Keyword", value="")
        channels = sorted(df["source_channel"].dropna().unique().tolist()) if "source_channel" in df.columns else []
        selected_channels = st.multiselect("Source Channel", options=channels, default=channels)
        selected_sentiments = st.multiselect("Sentiment", options=VALID_LABELS, default=VALID_LABELS)
        all_aspects = sorted(set(df_exploded["aspects"].dropna().tolist()))
        selected_aspects = st.multiselect("Aspects", options=all_aspects, default=all_aspects)

# Filter DataFrame Based on User Selections
df_view = df[df[LABEL_COL_FINAL].isin(selected_sentiments)].copy()
if selected_channels and "source_channel" in df_view.columns:
    df_view = df_view[df_view["source_channel"].isin(selected_channels)]

if keyword.strip():
    kw = keyword.strip().lower()
    text_col = "text" if "text" in df_view.columns else "clean_text"
    df_view = df_view[df_view[text_col].astype(str).str.lower().str.contains(kw, na=False) | 
                      df_view["clean_text"].astype(str).str.lower().str.contains(kw, na=False)]

df_view["aspects"] = df_view["clean_text"].astype(str).apply(tag_aspects)
df_view_exploded = df_view.explode("aspects").copy()
if selected_aspects:
    df_view_exploded = df_view_exploded[df_view_exploded["aspects"].isin(selected_aspects)]

# KPI Metrics
sent_counts = df_view[LABEL_COL_FINAL].value_counts()
sent_pct = df_view[LABEL_COL_FINAL].value_counts(normalize=True) * 100
net_sentiment = sent_pct.get("positive", 0) - sent_pct.get("negative", 0)

aspect_summary_tmp = make_aspect_summary(df_view_exploded)
top_aspects_for_kw = aspect_summary_tmp.head(3).index.tolist()
kw_map = top_keywords_per_aspect(df_view_exploded, top_aspects_for_kw, topn=4)

st.markdown(
    """
    <style>
      .kpi-grid {display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 15px 0;}
      .kpi-card {border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px; background: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
      .kpi-title {font-size: 13px; color: #7f8c8d; font-weight: 600; text-transform: uppercase;}
      .kpi-main {font-size: 26px; font-weight: 800; color: #2c3e50; margin: 5px 0;}
      .kpi-sub {font-size: 13px; color: #95a5a6;}
      .kpi-pill {background: #e8f4fd; color: #2980b9; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True
)

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card"><div class="kpi-title">Total Data</div><div class="kpi-main">{len(df_view):,}</div><div class="kpi-sub">Komentar Tervaluasi</div></div>
  <div class="kpi-card"><div class="kpi-title">Positif 😊</div><div class="kpi-main">{sent_counts.get("positive",0):,}</div><div class="kpi-sub">{sent_pct.get("positive",0):.1f}%</div></div>
  <div class="kpi-card"><div class="kpi-title">Negatif 😡</div><div class="kpi-main">{sent_counts.get("negative",0):,}</div><div class="kpi-sub">{sent_pct.get("negative",0):.1f}%</div></div>
  <div class="kpi-card"><div class="kpi-title">Netral 😐</div><div class="kpi-main">{sent_counts.get("neutral",0):,}</div><div class="kpi-sub">{sent_pct.get("neutral",0):.1f}%</div></div>
  <div class="kpi-card"><div class="kpi-title">Net Sentiment</div><div class="kpi-main">{net_sentiment:.1f}</div><div class="kpi-sub"><span class="kpi-pill">BHI Score</span></div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# TABS untuk berbagai analisis
tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Table", "📊 Aspect Analysis", "🔥 Cross-Channel", "🤖 Executive Report (AI)"])

# TAB 1: Data Table View
with tab1:
    st.subheader("Filtered Dataset")
    table_cols = [c for c in ["source_channel", "author", "text", LABEL_COL_FINAL] if c in df_view.columns]
    if table_cols:
        df_table = df_view[table_cols].rename(columns={"source_channel": "Channel", "author": "User", "text": "Komentar", LABEL_COL_FINAL: "Sentimen"})
        st.dataframe(df_table, use_container_width=True, height=400)
        st.download_button("⬇️ Download CSV", data=df_table.to_csv(index=False).encode("utf-8"), file_name="filtered_sentiment.csv", mime="text/csv")
    else:
        st.warning("Kolom tidak lengkap untuk ditampilkan.")

# TAB 2: Aspect Level Analysis
with tab2:
    st.subheader("Aspect-Level Distribution")
    aspect_summary = make_aspect_summary(df_view_exploded)
    aspect_dist = make_aspect_distribution(df_view_exploded)
    
    colA, colB = st.columns([1.5, 1])
    with colA:
        fig_zscore = fig_aspect_zscore_bar(aspect_summary)
        st.pyplot(fig_zscore)
    with colB:
        fig_matrix = fig_prioritization_matrix(df_view_exploded, min_mentions)
        st.pyplot(fig_matrix)
        
    st.markdown("#### Top Drivers")
    colC, colD = st.columns(2)
    fig_pain, fig_pos = fig_top_pain_positive(aspect_dist, min_mentions, topk)
    with colC: st.pyplot(fig_pain)
    with colD: st.pyplot(fig_pos)

# TAB 3: Cross Channel
with tab3:
    st.subheader("Channel Consistency")
    fig_heat = fig_cross_channel_heatmap(df_view_exploded, min_mentions)
    if fig_heat:
        st.pyplot(fig_heat)
    else:
        st.info("Data channel tidak mencukupi untuk heatmap.")

# TAB 4: GEMINI Report Generation & PDF Building
with tab4:
    st.subheader("Generate Strategic Report (PDF)")
    
    # Prompt Setup
    aspect_lines = []
    for aspect, row in aspect_summary.sort_values("z_score", ascending=False).iterrows():
        d = aspect_dist.loc[aspect] if aspect in aspect_dist.index else {"positive_rate":0, "negative_rate":0, "neutral_rate":0}
        aspect_lines.append(f"• {aspect:12} | Z:{row['z_score']:+4.2f} | Mnt:{int(row['total_mentions']):3} | Pos:{d.get('positive_rate',0)*100:4.1f}% | Neg:{d.get('negative_rate',0)*100:4.1f}%")

    top_complaints = "\n".join([f"- {idx} (Z: {row['z_score']:.2f})" for idx, row in aspect_summary[aspect_summary["total_mentions"] >= min_mentions].sort_values("z_score").head(3).iterrows()])
    top_praises = "\n".join([f"- {idx} (Z: {row['z_score']:.2f})" for idx, row in aspect_summary[aspect_summary["total_mentions"] >= min_mentions].sort_values("z_score", ascending=False).head(3).iterrows()])

    # PROMPT Engineering untuk Gemini
    prompt = f"""
Anda adalah Senior Strategy Consultant McKinsey/BCG untuk Xiaomi Global Market Intelligence Team. 
Tugas Anda: Buat Strategic Intelligence Report berkualitas eksekutif dari data YouTube.

TARGET: {TARGET_PRODUCT}
Total Sampel: {len(df_view):,} | Net Sentiment: {net_sentiment:.1f}/100

ASPECT BREAKDOWN:
{chr(10).join(aspect_lines)}

STRENGTHS (Z > 1.5):
{top_praises}

WEAKNESSES (Z < -1.5):
{top_complaints}

INSTRUKSI PENULISAN (PENTING UNTUK PDF RENDER):
1. Mulai langsung dengan judul: "### STRATEGIC INTELLIGENCE REPORT"
2. Gunakan HANYA format Bullet Points (-) atau Numbering (1. 2.) untuk daftar/rencana.
3. DILARANG KERAS menggunakan format Tabel Markdown (| Kolom | Kolom |) karena akan merusak format PDF PDFReportLab.
4. Gunakan gaya bahasa Konsultan Strategis formal.

STRUKTUR WAJIB:
### 1. EXECUTIVE SUMMARY & ADVISORY
(Tulis Status: STRENGTHEN/PIVOT/RECALL, narasi utama, dan 3 rekomendasi)

### 2. DEEP-DIVE MARKET DIAGNOSIS
(Bahas Competitive Moat & Churn Analysis ke brand lain)

### 3. PRODUCT & ECOSYSTEM IMPLICATIONS
(Hardware issue vs Software issue)

### 4. 90-DAY ACTIONABLE ROADMAP
(Gunakan bullet point berjenjang untuk membagi timeline Bulan 1, Bulan 2, Bulan 3. Ingat: Jangan pakai tabel).
    """.strip()

    if "report_text" not in st.session_state: st.session_state["report_text"] = ""
    if "pdf_ready" not in st.session_state: st.session_state["pdf_ready"] = False

    if st.button("🚀 Generate AI Report & Build PDF", type="primary"):
        with st.spinner(f"Analisis data menggunakan {MODEL_NAME}..."):
            try:
                st.session_state["report_text"] = generate_report_text(prompt)
                build_pdf_from_text(st.session_state["report_text"], PDF_PATH)
                st.session_state["pdf_ready"] = True
                st.success("✅ Report & PDF berhasil di-generate!")
            except Exception as e:
                st.error(f"Gagal generate report: {e}")

    if st.session_state["report_text"]:
        st.text_area("Preview Laporan AI", value=st.session_state["report_text"], height=300)

    if st.session_state["pdf_ready"] and PDF_PATH.exists():
        st.download_button(
            label="📄 Download Executive Report (.PDF)",
            data=PDF_PATH.read_bytes(),
            file_name=f"Report_{TARGET_PRODUCT.replace(' ','_')}.pdf",
            mime="application/pdf"
        )
