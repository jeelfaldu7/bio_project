import streamlit as st
import pandas as pd
import json

from src.agents.fetcher import PubMedFetcher

fetcher = PubMedFetcher(rss_urls=rss_feeds)
articles = fetcher.fetch()

print(f"Fetched {len(articles)} articles.")
