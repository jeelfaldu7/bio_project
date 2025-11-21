import streamlit as st
import pandas as pd
import json
from collections import Counter
import altair as alt

# Load articles
with open("data/rss_summarized.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

df = pd.DataFrame(articles)

st.title("Biotech News & Trend Analyzer")

# Filter by keyword
keyword = st.text_input("Search articles by keyword")
if keyword:
    df = df[df['summary_text'].str.contains(keyword, case=False) | df['title'].str.contains(keyword, case=False)]

# Show table
st.dataframe(df[['title', 'ai_summary', 'published', 'link']])




# --PLOTS--
# Example trend chart: number of articles per month
df['published_date'] = pd.to_datetime(df['published'])
trend = df.groupby(df['published_date'].dt.to_period('M')).size()
st.line_chart(trend)





# Pie chart: keyword frequency in summaries
st.subheader("Keyword Frequency in Summaries")

# Simple keyword extraction (split by space, lowercase)
all_text = " ".join(df['summary_text'].tolist()).lower()
words = all_text.split()

# Optionally filter out common stopwords
stopwords = {"the", "and", "of", "in", "for", "to", "with", "on", "a", "an", "at", "as", "by", "is", "are", "this", "that"}
filtered_words = [w.strip(".,()") for w in words if w not in stopwords]

# Count top 10 words
word_counts = Counter(filtered_words)
top_words = word_counts.most_common(10)
top_df = pd.DataFrame(top_words, columns=["keyword", "count"])

# Display pie chart
pie_chart = alt.Chart(top_df).mark_arc().encode(
    theta=alt.Theta(field="count", type="quantitative"),
    color=alt.Color(field="keyword", type="nominal"),
    tooltip=["keyword", "count"]
)
st.altair_chart(pie_chart, use_container_width=True)