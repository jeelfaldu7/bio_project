# main.py — rewritten with unified dark theme
import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from datetime import datetime
from dateutil import parser
from collections import Counter
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go

# -------------------------
# 1) PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Biotech Trend Intelligence",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# -------------------------
# 2) COLORS
# -------------------------
DASHBOARD_BG = "#00091a"
SURFACE_BG = "#001022"
TEXT_COLOR = "white"
ACCENT_COLOR = "#4dc4ff"

# -------------------------
# 3) DARK PLOTLY TEMPLATE (Unified)
# -------------------------
pio.templates["biotech_dark"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=DASHBOARD_BG,
        plot_bgcolor=DASHBOARD_BG,
        font=dict(color=TEXT_COLOR),
        title=dict(font=dict(color=TEXT_COLOR)),
        xaxis=dict(showgrid=False, color=TEXT_COLOR),
        yaxis=dict(showgrid=False, color=TEXT_COLOR),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR)),
        colorway=[ACCENT_COLOR, "#ff9f43", "#ff6b81", "#6c5ce7", "#00d2d3"]
    )
)
pio.templates.default = "biotech_dark"

# -------------------------
# 4) CSS
# -------------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="stApp"] {{
        font-family: 'Inter', sans-serif;
        background: {DASHBOARD_BG};
        color: #f4f7fb;
    }}

    .big-title {{
        font-size: 48px;
        font-weight: 700;
        background: linear-gradient(90deg, {ACCENT_COLOR}, {TEXT_COLOR});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }}

    .subtle {{
        text-align: center;
        font-size: 18px;
        color: #c9d3ea;
    }}

    .plot-container > div {{
        background: rgba(255,255,255,0.04) !important;
        border-radius: 18px;
        padding: 18px;
    }}
</style>
""", unsafe_allow_html=True)

# -------------------------
# 5) BANNER
# -------------------------
banner = Image.open("thumbnail.png")
st.image(banner, use_column_width=True)

# -------------------------
# 6) LOAD TRENDING DATA
# -------------------------
@st.cache_data
def load_trending_json(path="data/trending_topics.json"):
    p = Path(path)
    if not p.exists():
        return []
    return json.load(p.open("r", encoding="utf-8"))

raw = load_trending_json()
if not raw:
    st.error("Missing data/trending_topics.json")
    st.stop()

rows = []
for r in raw:
    rows.append({
        "topic": r.get("topic") or r.get("title") or "Unknown",
        "trend_score": float(r.get("trend_score") or r.get("score") or np.nan),
        "articles": r.get("articles") or [],
        "key_terms": r.get("key_terms") or [],
        "summary": r.get("summary") or r.get("ai_summary") or ""
    })

df = pd.DataFrame(rows)
df["articles_str"] = df["articles"].apply(lambda x: "• " + "• ".join([a.get("title") if isinstance(a, dict) else str(a) for a in x]) if isinstance(x, list) else str(x))
df["key_terms_str"] = df["key_terms"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))

# -------------------------
# 7) LOAD ORIGINAL RAW DATA
# -------------------------
with open("data/rss_summarized.json", "r", encoding="utf-8") as f:
    original_json = json.load(f)

# -------------------------
# 8) DATE PARSING
# -------------------------
def parse_published_date(date_str):
    if not isinstance(date_str, str):
        return None
    try:
        return pd.to_datetime(date_str, format="%b %d, %Y %I:%M%p")
    except:
        pass
    try:
        tzinfos = {"EST": -5*3600, "EDT": -4*3600, "PST": -8*3600, "PDT": -7*3600}
        return parser.parse(date_str, tzinfos=tzinfos)
    except:
        return pd.to_datetime(date_str, errors="coerce")

def restore_published_dates_flat(original_json):
    rows = []
    for art in original_json:
        dt = parse_published_date(art.get("published") or art.get("published_at"))
        if dt is not None:
            if dt.tzinfo is None:
                dt = pd.Timestamp(dt).tz_localize("UTC")
            else:
                dt = pd.Timestamp(dt).tz_convert("UTC")
        rows.append({
            "article_title": art.get("title", ""),
            "source": art.get("source", "Unknown"),
            "published_dt": dt
        })
    flat = pd.DataFrame(rows).dropna(subset=["published_dt"])
    return flat

flat_df = restore_published_dates_flat(original_json)
flat_df["topic"] = ""
for i in range(min(len(flat_df), len(rows))):
    flat_df.at[i, "topic"] = rows[i]["topic"]

# -------------------------
# 9) NORMALIZE TREND SCORE
# -------------------------
df["trend_score_norm"] = (
    (df["trend_score"] - df["trend_score"].min()) /
    (df["trend_score"].max() - df["trend_score"].min())
) * 100

# -------------------------
# 10) SIDEBAR FILTERS
# -------------------------
st.sidebar.markdown("<h2>Filters</h2>", unsafe_allow_html=True)
score_range = st.sidebar.slider("Trend Score", 0, 100, (0, 100))
search_q = st.sidebar.text_input("Search topic or summary")
all_terms = sorted({t for terms in df.key_terms if isinstance(terms, list) for t in terms})
selected_terms = st.sidebar.multiselect("Key Terms", all_terms)

filtered = df[
    (df.trend_score_norm >= score_range[0]) &
    (df.trend_score_norm <= score_range[1])
]
if search_q:
    q = search_q.lower()
    filtered = filtered[
        filtered.topic.str.lower().str.contains(q) |
        filtered.summary.str.lower().str.contains(q)
    ]
if selected_terms:
    filtered = filtered[
        filtered.key_terms.apply(lambda terms: any(t in terms for t in selected_terms))
    ]

# -------------------------
# 11) METRICS
# -------------------------
st.metric("Topics", len(filtered))

# -------------------------
# 12) CHARTS
# -------------------------
st.subheader("📈 Top Topics by Trend Score")

if not filtered.empty:
    fig = px.bar(
        filtered.sort_values("trend_score", ascending=False).head(25),
        x="trend_score",
        y="topic",
        orientation="h",
        hover_data=["summary", "key_terms_str"],
        template="biotech_dark"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No topics match the filters.")

# -------------------------
# TIME SERIES
# -------------------------
st.subheader("🕒 Trend Activity Over Time")
ts_df = flat_df[flat_df.topic.isin(filtered.topic)]
if not ts_df.empty:
    ts_df["date"] = ts_df.published_dt.dt.floor("D")
    agg = ts_df.groupby("date").size().reset_index(name="count")
    fig_ts = px.line(agg, x="date", y="count", markers=True, template="biotech_dark")
    st.plotly_chart(fig_ts, use_container_width=True)
else:
    st.info("No matching articles.")

# -------------------------
# COMPANIES
# -------------------------
st.subheader("🏢 Top Companies / Entities Mentioned")
company_counter = Counter(t.title() for terms in filtered.key_terms for t in terms)
company_df = pd.DataFrame(company_counter.most_common(20), columns=["company", "count"])
if not company_df.empty:
    fig_comp = px.bar(company_df, x="count", y="company", orientation="h", template="biotech_dark")
    st.plotly_chart(fig_comp, use_container_width=True)
else:
    st.info("No terms detected.")

# -------------------------
# HEATMAP
# -------------------------
st.subheader("📊 Cluster Occurrence Across Sources")
flat_filtered = flat_df[flat_df.topic.isin(filtered.topic)]
if not flat_filtered.empty:
    flat_filtered["cluster"] = flat_filtered["topic"].replace("", "Unknown")
    pivot = flat_filtered.pivot_table(index="cluster", columns="source", values="article_title", aggfunc="count", fill_value=0)
    fig_heat = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="YlGnBu"))
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("Not enough data.")

# -------------------------
# WORD CLOUD
# -------------------------
st.subheader("💡 Trending Concepts")
concepts = [t for terms in filtered.key_terms for t in terms]
if concepts:
    text = " ".join(concepts)
    wc = WordCloud(width=800, height=400, background_color=DASHBOARD_BG, colormap="Blues", max_words=60).generate(text)
    fig, ax = plt.subplots(figsize=(12,5))
    fig.patch.set_facecolor(DASHBOARD_BG)
    ax.set_facecolor(DASHBOARD_BG)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.info("No concepts available.")

# -------------------------
# EXPANDERS
# -------------------------
st.subheader("📚 Detailed Trend Breakdown")
for _, row in filtered.iterrows():
    with st.expander(f"{row['topic']} — Score: {row['trend_score_norm']:.0f}"):
        st.markdown("### Summary")
        st.write(row.summary)
        st.markdown("### Articles")
        st.write(row.articles_str)
        st.markdown("### Key Terms")
        st.write(
            " ".join([f"<span class='chip'>{t}</span>" for t in row.key_terms]),
            unsafe_allow_html=True
        )
