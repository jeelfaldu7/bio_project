import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from datetime import datetime
from collections import Counter
from pathlib import Path

# ----------------------
# Page config + CSS
# ----------------------
st.set_page_config(page_title="Biotech Trend Intelligence", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
        .stApp { background-color: #fafafa; }
        .big-title { font-size: 38px; font-weight:700; 
                      background: -webkit-linear-gradient(90deg,#0061ff,#60efff);
                      -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .subtle { font-size:16px; color:#666; margin-top:-8px; }
        .chip { display:inline-block; padding:4px 10px; margin:3px; background:#e6f0ff; color:#003d99; border-radius:12px; font-size:12px; font-weight:500; }
        .metric-card { padding:16px; border-radius:12px; background:white; box-shadow:0 1px 6px rgba(0,0,0,0.06); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------
# Helper functions
# ----------------------
@st.cache_data
def load_trending_json(path: str = "data/trending_topics.json"):
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def safe_to_datetime(x):
    if not x or not isinstance(x, str):
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%d"):
        try:
            return datetime.fromisoformat(x) if "T" in x else datetime.strptime(x, fmt)
        except Exception:
            continue
    try:
        return pd.to_datetime(x, utc=False)
    except Exception:
        return None

# ----------------------
# Load data
# ----------------------
raw = load_trending_json("data/trending_topics.json")
if not raw:
    st.error("No `data/trending_topics.json` found. Please run the pipeline and place the JSON at data/trending_topics.json")
    st.stop()

# Convert to DataFrame
rows = []
flattened_rows = []
for r in raw:
    topic = r.get("topic") or r.get("title") or "Unknown"
    score = r.get("trend_score") or r.get("score") or None
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
    
    # Flatten for per-article info
    for art in articles:
        if isinstance(art, dict):
            title = art.get("title") or art.get("headline") or ""
            source = art.get("source") or art.get("source_name") or ""
            published = art.get("published") or art.get("published_at") or art.get("fetched_at") or None
        else:
            title = str(art)
            source = ""
            published = None
        flattened_rows.append({"topic": topic, "article_title": title, "source": source, "published": published})

df = pd.DataFrame(rows)
df["articles_str"] = df["articles"].apply(lambda x: "• " + "\n• ".join(x) if isinstance(x, list) else str(x))
df["key_terms_str"] = df["key_terms"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))

flat_df = pd.DataFrame(flattened_rows)
if not flat_df.empty:
    flat_df["published_dt"] = flat_df["published"].apply(lambda x: safe_to_datetime(x))
else:
    flat_df["published_dt"] = pd.Series(dtype="datetime64[ns]")

# ----------------------
# Sidebar filters
# ----------------------
st.sidebar.header("Filters")
score_min = float(df.trend_score.min(skipna=True)) if df.trend_score.notna().any() else 0.0
score_max = float(df.trend_score.max(skipna=True)) if df.trend_score.notna().any() else 10.0
score_range = st.sidebar.slider("Trend Score", min_value=score_min, max_value=score_max, value=(score_min, score_max))
search_q = st.sidebar.text_input("Search topic or summary")
all_terms = sorted({t for terms in df["key_terms"] for t in (terms if isinstance(terms, list) else [])})
selected_terms = st.sidebar.multiselect("Key terms (filter)", all_terms)
min_articles = st.sidebar.slider("Min # of articles per topic", min_value=0, max_value=50, value=0)

filtered = df[
    (df["trend_score"].fillna(-999) >= score_range[0]) &
    (df["trend_score"].fillna(-999) <= score_range[1])
].copy()

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

# ----------------------
# Header + metrics
# ----------------------
st.markdown('<p class="big-title">Biotech Trend Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="subtle">Portfolio dashboard — clusters, trend intensity, and signal sources.</p>', unsafe_allow_html=True)
st.divider()

col1, col2, col3 = st.columns([3, 1, 1])
with col2: st.metric("Topics", len(filtered))
with col3: st.metric("Avg Score", round(filtered.trend_score.mean(skipna=True), 2) if len(filtered)>0 else 0)
st.divider()

# ----------------------
# Charts (stacked vertically)
# ----------------------
# 1. Trend Score Bar
st.subheader("📈 Top Topics by Trend Score")
display_df = filtered.sort_values("trend_score", ascending=False).head(25)
if not display_df.empty:
    fig = px.bar(display_df, x="trend_score", y="topic", orientation="h",
                 hover_data=["summary", "key_terms_str"], labels={"trend_score":"Trend Score","topic":"Topic"})
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No topics match the filters.")
st.divider()

# 2. Time Series Articles per Day
st.subheader("🕒 Articles Over Time")
if not flat_df.empty and flat_df["published_dt"].notna().any():
    ts_df = flat_df.dropna(subset=["published_dt"]).copy()
    ts_df["date"] = pd.to_datetime(ts_df["published_dt"]).dt.date
    ts_agg = ts_df.groupby("date").size().reset_index(name="count")
    ts_agg = ts_agg.sort_values("date")
    fig_ts = px.line(ts_agg, x="date", y="count", title="Articles per Day", markers=True)
    fig_ts.update_layout(plot_bgcolor="white", paper_bgcolor="white", xaxis_title="Date", yaxis_title="Article Count")
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
            if len(t)>2: company_counter[t.title()] += 1
company_df = pd.DataFrame(company_counter.most_common(20), columns=["company","count"])
if not company_df.empty:
    fig_comp = px.bar(company_df, x="count", y="company", orientation="h")
    fig_comp.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_comp, use_container_width=True)
else:
    st.info("No company-like terms detected.")

st.divider()

# 4. Heatmap of Clusters across Sources
st.subheader("📊 Cluster Occurrence Across Sources")
if not flat_df.empty and "source" in flat_df.columns:
    heat_df = flat_df.copy()
    heat_df["cluster"] = heat_df["topic"]
    pivot = heat_df.pivot_table(index="cluster", columns="source", values="article_title", aggfunc="count", fill_value=0)
    if not pivot.empty:
        fig_heat = px.imshow(pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                             labels=dict(x="Source", y="Cluster", color="Article Count"),
                             color_continuous_scale="YlGnBu")
        fig_heat.update_layout(height=500)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Not enough data for heatmap.")
else:
    st.info("No per-article source data available for heatmap.")

st.divider()

# 5. WordCloud Concepts
st.subheader("💡 Trending Concepts (WordCloud)")
all_concepts = []
for terms in filtered["key_terms"]:
    if isinstance(terms, list):
        all_concepts.extend(terms)
if all_concepts:
    text = " ".join(all_concepts)
    wc = WordCloud(width=800, height=400, background_color="white", collocations=False).generate(text)
    fig, ax = plt.subplots(figsize=(12,5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.info("No concepts available to generate WordCloud.")

st.divider()

# ----------------------
# Expandable trend cards
# ----------------------
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
