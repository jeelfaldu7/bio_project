import streamlit as st
import pandas as pd
import json
import plotly.express as px

# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------
with open("data/trending_topics.json", "r") as f:
    trends = json.load(f)

df = pd.DataFrame(trends)

# Format lists for display
df["articles_str"] = df["articles"].apply(lambda x: "• " + "\n• ".join(x))
df["key_terms_str"] = df["key_terms"].apply(lambda x: ", ".join(x))

# ------------------------------------------------------
# PAGE CONFIG + CUSTOM CSS
# ------------------------------------------------------
st.set_page_config(
    page_title="Biotech Trend Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium styling
st.markdown("""
    <style>
        .main {
            background-color: #fafafa;
        }
        .big-title {
            font-size: 40px;
            font-weight: 700;
            margin-bottom: -10px;
            background: -webkit-linear-gradient(90deg, #0061ff, #60efff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtle {
            font-size: 18px;
            color: #666;
        }
        .chip {
            display: inline-block;
            padding: 4px 10px;
            margin: 2px;
            background: #e6f0ff;
            color: #003d99;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }
        .metric-card {
            padding: 18px;
            border-radius: 14px;
            background: white;
            box-shadow: 0px 1px 4px rgba(0,0,0,0.08);
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# HEADER
# ------------------------------------------------------
st.markdown('<p class="big-title">Biotech Trend Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="subtle">A portfolio-ready dashboard analyzing macro trends in biopharma, R&D innovation, and regulatory activity.</p>', unsafe_allow_html=True)

st.divider()

# ------------------------------------------------------
# SIDEBAR — ADVANCED FILTERS
# ------------------------------------------------------
st.sidebar.header("🔍 Filters")

# Search box
search_query = st.sidebar.text_input("Search Topics or Summaries")

# Trend score range
score_range = st.sidebar.slider(
    "Trend Score Range",
    min_value=int(df.trend_score.min()),
    max_value=int(df.trend_score.max()),
    value=(7, 10)
)

# Key term filtering
all_terms = sorted({term for terms in df.key_terms for term in terms})
selected_terms = st.sidebar.multiselect("Filter by Key Terms", all_terms)

# Apply filters
filtered = df[
    (df.trend_score >= score_range[0]) &
    (df.trend_score <= score_range[1])
]

if search_query:
    q = search_query.lower()
    filtered = filtered[
        filtered.topic.str.lower().str.contains(q) |
        filtered.summary.str.lower().str.contains(q)
    ]

if selected_terms:
    filtered = filtered[filtered.key_terms.apply(lambda terms: any(t in terms for t in selected_terms))]

# ------------------------------------------------------
# METRIC CARDS
# ------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Trends", len(filtered))
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Average Score", round(filtered.trend_score.mean(), 2))
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Max Score", int(filtered.trend_score.max()))
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ------------------------------------------------------
# PLOTLY BAR CHART
# ------------------------------------------------------
st.subheader("📈 Trend Score Distribution (Interactive)")

chart_df = filtered.sort_values("trend_score", ascending=False)

fig = px.bar(
    chart_df,
    x="topic",
    y="trend_score",
    title="Trend Scores by Topic",
    hover_data=["summary", "key_terms_str"],
)

fig.update_layout(
    xaxis_title="Topic",
    yaxis_title="Score",
    title_x=0.2,
    plot_bgcolor="white",
    paper_bgcolor="white",
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ------------------------------------------------------
# EXPANDABLE TREND CARDS
# ------------------------------------------------------
st.subheader("📚 Detailed Trend Breakdown")

for _, row in filtered.iterrows():
    with st.expander(f"{row['topic']}  —  Score: {row['trend_score']}"):
        st.markdown(f"### 🧠 Summary")
        st.write(row["summary"])

        st.markdown("### 📰 Articles")
        st.markdown(row["articles_str"])

        st.markdown("### 🔑 Key Terms")
        for term in row["key_terms"]:
            st.markdown(f"<span class='chip'>{term}</span>", unsafe_allow_html=True)
        st.write("")