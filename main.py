# main.py
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
from collections import Counter
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go

# -------------------------
# 1) PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Biotech Trend Dashboard",
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
LIGHT_BG = "#0c1a3d"

# -------------------------
# 3) CREATE PLOTLY TEMPLATE
# -------------------------
pio.templates["biotech_dark"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=LIGHT_BG,
        plot_bgcolor=LIGHT_BG,
        font=dict(color=TEXT_COLOR),
        title=dict(font=dict(color=TEXT_COLOR)),
        xaxis=dict(showgrid=False, color=TEXT_COLOR),
        yaxis=dict(showgrid=False, color=TEXT_COLOR),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR)),
        colorway=[ACCENT_COLOR, "#ff9f43", "#ff6b81", "#6c5ce7", "#00d2d3"]
    ),
    data={
        "scatter": [go.Scatter()],
        "bar": [go.Bar()],
        "heatmap": [go.Heatmap()],
        "pie": [go.Pie()],
        "histogram": [go.Histogram()],
        "box": [go.Box()],
        "violin": [go.Violin()]
    }
)

#pio.templates.default = "biotech_dark"

# Helper function to enforce template on existing figure
def apply_dark_theme(fig):
    return fig

# -------------------------
# 4) CSS FOR STREAMLIT PAGE
# -------------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Page and content */
    html, body, [class*="stApp"] {{
        font-family: 'Inter', sans-serif;
        background: {DASHBOARD_BG};
        color: #f4f7fb;
    }}

    /* Titles */
    .big-title {{
        font-size: 48px;
        font-weight: 700;
        background: linear-gradient(90deg, {ACCENT_COLOR}, {TEXT_COLOR});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 4px;
    }}

    .subtle {{
        text-align: center;
        font-size: 18px;
        color: #c9d3ea;
        margin-top: -8px;
        margin-bottom: 18px;
    }}

    /* Metric cards */
    .stMetric {{
        background: rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 12px 14px;
        backdrop-filter: blur(6px);
    }}

    /* Chips */
    .chip {{
        display: inline-block;
        padding: 6px 12px;
        margin: 4px 6px 4px 0;
        background: rgba(255,255,255,0.12);
        color: {ACCENT_COLOR};
        border-radius: 14px;
        font-size: 13px;
        font-weight: 500;
        letter-spacing: .2px;
    }}

    /* Expander */
    details {{
        background: rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 8px 16px;
        margin-bottom: 10px;
        border: 1px solid rgba(255,255,255,0.08);
    }}
    summary {{
        font-size: 18px;
        font-weight: 600;
        padding: 6px;
    }}

    /* Plot container */
    .plot-container > div {{
        background: rgba(255,255,255,0.06) !important;
        border-radius: 18px;
        padding: 18px;
    }}

    /* Headings */
    h1, h2, h3, h4, h5, h6 {{
        color: white !important;
    }}
</style>
""", unsafe_allow_html=True)

# -------------------------
# 5) HERO BANNER
# -------------------------
banner = Image.open("thumbnail.png")
st.image(banner)

# -------------------------
# 6) HELPER FUNCTIONS
# -------------------------
@st.cache_data
def load_trending_json(path: str = "data/trending_topics.json"):
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def safe_to_datetime(x):
    try:
        return pd.to_datetime(x)
    except:
        return None

# -------------------------
# 7) LOAD DATA
# -------------------------
raw = load_trending_json()
if not raw:
    st.error("No `data/trending_topics.json` found.")
    st.stop()

rows = []
flattened_rows = []
for r in raw:
    topic = r.get("topic") or r.get("title") or "Unknown"
    score = r.get("trend_score") or r.get("score") or np.nan
    articles = r.get("articles") or []
    key_terms = r.get("key_terms") or []
    summary = r.get("summary") or r.get("ai_summary") or ""
    
    rows.append({
        "topic": topic,
        "trend_score": float(score) if score is not None else np.nan,
        "articles": articles,
        "key_terms": key_terms,
        "summary": summary
    })
    
    for art in articles:
        if isinstance(art, dict):
            title = art.get("title") or art.get("headline") or ""
            source = art.get("source") or art.get("source_name") or ""
            published = art.get("published") or art.get("published_at") or None
        else:
            title = str(art)
            source = ""
            published = None
        flattened_rows.append({"topic": topic, "article_title": title, "source": source, "published": published})

df = pd.DataFrame(rows)
df["articles_str"] = df["articles"].apply(lambda x: "• " + "\n• ".join(x) if isinstance(x, list) else str(x))
df["key_terms_str"] = df["key_terms"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))


# Load published dates from original JSON
# -------------------------
# Load your processed filtered topics (example)
# filtered_df should have columns: 'topic' and 'articles' (list of article titles or dicts)
# -------------------------
# filtered_df = pd.read_csv("filtered_topics.csv")  # or however you have it

# -------------------------
# Load original JSON (with published dates)
# -------------------------
with open("data/rss_summarized.json", "r", encoding="utf-8") as f:
    original_json = json.load(f)

# -------------------------
# Function to restore published dates
# -------------------------
def restore_published_dates(filtered_df, original_json):
    restored_flat_rows = []

    for _, row in filtered_df.iterrows():
        topic = row['topic']
        articles = row['articles']  # list of article titles or dicts

        # Find the original topic entry
        orig_topic = next((r for r in original_json if r.get("title") == topic or r.get("topic") == topic), None)

        original_articles = orig_topic.get("articles", []) if orig_topic else []

        for i, art in enumerate(articles):
            # Use matching original article by index
            orig_art = original_articles[i] if i < len(original_articles) else {}

            # Extract title
            title = art.get("title") if isinstance(art, dict) else str(art)

            # Extract published string
            published_str = None
            if isinstance(orig_art, dict):
                published_str = orig_art.get("published") or orig_art.get("published_at")

            # Parse datetime
            published_dt = None
            if published_str:
                try:
                    published_dt = datetime.strptime(published_str, "%b %d, %Y %I:%M%p")
                except:
                    published_dt = None

            restored_flat_rows.append({
                "topic": topic,
                "article_title": title,
                "source": orig_art.get("source") if isinstance(orig_art, dict) else "",
                "published": published_str,
                "published_dt": published_dt
            })

    return pd.DataFrame(restored_flat_rows)

# -------------------------
# Restore flat dataframe with published dates
# -------------------------
flat_df = restore_published_dates(df, original_json)
flat_df["published_dt"] = pd.to_datetime(flat_df["published"], errors="coerce")
flat_df = flat_df.dropna(subset=["published_dt"]).copy()

# -------------------------
# 8) SIDEBAR FILTERS
# -------------------------
st.sidebar.header("Filters")
score_min, score_max = float(df.trend_score.min(skipna=True)), float(df.trend_score.max(skipna=True))
score_range = st.sidebar.slider("Trend Score", min_value=score_min, max_value=score_max, value=(score_min, score_max))
search_q = st.sidebar.text_input("Search topic or summary")
all_terms = sorted({t for terms in df["key_terms"] for t in (terms if isinstance(terms, list) else [])})
selected_terms = st.sidebar.multiselect("Key terms (filter)", all_terms)
min_articles = st.sidebar.slider("Min # of articles per topic", 0, 50, 0)

filtered = df[
    (df["trend_score"].fillna(-999) >= score_range[0]) &
    (df["trend_score"].fillna(-999) <= score_range[1])
]

if search_q:
    q = search_q.lower()
    filtered = filtered[
        filtered["topic"].str.lower().str.contains(q, na=False) |
        filtered["summary"].str.lower().str.contains(q, na=False)
    ]
if selected_terms:
    filtered = filtered[
        filtered["key_terms"].apply(lambda terms: any(t in terms for t in selected_terms) if isinstance(terms, list) else False)
    ]
if min_articles > 0:
    filtered = filtered[filtered["articles"].apply(lambda arr: len(arr) if isinstance(arr, list) else 0) >= min_articles]

# -------------------------
# 9) HEADER + METRICS
# -------------------------
st.markdown('<p class="subtle">Interactive dashboard — clusters, trend intensity, and signal sources.</p>', unsafe_allow_html=True)
st.divider()

sp1, center_cols, sp2 = st.columns([1,3,1])

with center_cols:
    c1, c2 = st.columns(2)
    with c1: 
        st.metric("Topics", len(filtered))
    with c2: 
        st.metric("Avg Score", round(filtered.trend_score.mean(skipna=True), 2) if len(filtered)>0 else 0)

st.divider()

# -------------------------
# 10) CHARTS
# -------------------------
# 1. Trend Score Bar
st.subheader("📈 Top Topics by Trend Score")
display_df = filtered.sort_values("trend_score", ascending=False).head(25)
if not display_df.empty:
    fig = px.bar(
        display_df,
        x="trend_score",
        y="topic",
        orientation="h",
        hover_data=["summary", "key_terms_str"],
        labels={"trend_score": "Trend Score", "topic": "Topic"},
        template="biotech_dark"  # use your dark template
    )
    fig.update_layout(
        title="",
        paper_bgcolor=LIGHT_BG,
        plot_bgcolor=LIGHT_BG,
        xaxis=dict(showgrid=False, color=TEXT_COLOR),
        yaxis=dict(showgrid=False, color=TEXT_COLOR),
        title_font=dict(color=TEXT_COLOR),
        template="biotech_dark"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No topics match the filters.")

# 2. Time Series Articles per Day
st.subheader("🕒 Articles Over Time")

if not flat_df.empty and flat_df["published_dt"].notna().any():
    # Ensure datetime dtype
    ts_df = flat_df.dropna(subset=["published_dt"]).copy()
    ts_df["date"] = pd.to_datetime(ts_df["published_dt"])  # keep full datetime
    
    # Aggregate per day
    ts_agg = ts_df.groupby(ts_df["date"].dt.date).size().reset_index(name="count")
    ts_agg["date"] = pd.to_datetime(ts_agg["date"])  # convert back to datetime for Plotly
    
    # Plot
    fig_ts = px.line(
        ts_agg,
        x="date",
        y="count",
        title="Articles per Day",
        markers=True,
        template="biotech_dark"
    )
    fig_ts.update_layout(
        paper_bgcolor=LIGHT_BG,
        plot_bgcolor=LIGHT_BG,
        xaxis=dict(showgrid=False, color=TEXT_COLOR),
        yaxis=dict(showgrid=False, color=TEXT_COLOR),
        title_font=dict(color=TEXT_COLOR),
    )
    st.plotly_chart(fig_ts, use_container_width=True)
else:
    st.info("No article publication dates available.")

st.divider()

# 3. Top Companies
st.subheader("🏢 Top Companies / Entities Mentioned")
company_counter = Counter()
for terms in filtered["key_terms"]:
    if isinstance(terms, list):
        for t in terms:
            if len(t) > 2:
                company_counter[t.title()] += 1
company_df = pd.DataFrame(company_counter.most_common(20), columns=["company", "count"])
if not company_df.empty:
    fig_comp = px.bar(
        company_df,
        x="count",
        y="company",
        orientation="h",
        template="biotech_dark"
    )
    fig_comp.update_layout(
        title="",
        paper_bgcolor=LIGHT_BG,
        plot_bgcolor=LIGHT_BG,
        xaxis=dict(showgrid=False, color=TEXT_COLOR),
        yaxis=dict(showgrid=False, color=TEXT_COLOR),
        title_font=dict(color=TEXT_COLOR),
        yaxis_categoryorder="total ascending",
        template="biotech_dark"
    )
    st.plotly_chart(fig_comp, use_container_width=True)
else:
    st.info("No company-like terms detected.")

# 4. Heatmap of Clusters across Sources
st.subheader("📊 Cluster Occurrence Across Sources")
if not flat_df.empty and "source" in flat_df.columns:
    heat_df = flat_df.copy()
    heat_df["cluster"] = heat_df["topic"]
    pivot = heat_df.pivot_table(
        index="cluster",
        columns="source",
        values="article_title",
        aggfunc="count",
        fill_value=0
    )
    if not pivot.empty:
        fig_heat = px.imshow(
            pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            labels=dict(x="Source", y="Cluster", color="Article Count"),
            color_continuous_scale="YlGnBu",
            template="biotech_dark"
        )
        fig_heat.update_layout(
            title="",
            paper_bgcolor=LIGHT_BG,
            plot_bgcolor=LIGHT_BG,
            xaxis=dict(showgrid=False, color=TEXT_COLOR),
            yaxis=dict(showgrid=False, color=TEXT_COLOR),
            title_font=dict(color=TEXT_COLOR),
            height=500,
            template="biotech_dark"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Not enough data for heatmap.")
else:
    st.info("No per-article source data available for heatmap.")

# 5. WordCloud Concepts
st.subheader("💡 Trending Concepts (WordCloud)")
all_concepts = [t for terms in filtered["key_terms"] if isinstance(terms, list) for t in terms]
if all_concepts:
    text = " ".join(all_concepts)
    wc = WordCloud(width=800, height=400, background_color=LIGHT_BG, colormap="Blues", collocations=False).generate(text)
    fig, ax = plt.subplots(figsize=(12,5))
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.info("No concepts available to generate WordCloud.")
st.divider()

# =========================================================
# Expandable Trend Cards
# =========================================================
st.subheader("📚 Detailed Trend Breakdown")
for _, row in filtered.iterrows():
    with st.expander(f"{row['topic']}  —  Score: {row['trend_score']}"):
        st.markdown("### 🧠 Summary")
        st.write(row["summary"])
        st.markdown("### 📰 Articles")
        st.markdown(row["articles_str"])
        st.markdown("### 🔑 Key Terms")
        for term in row["key_terms"]:
            st.markdown(f"<span class='chip'>{term}</span>", unsafe_allow_html=True)
        st.write("")