from src.agents.fetcher import PubMedFetcher

# Example RSS feeds (you will replace these with real PubMed RSS URLs)
rss_feeds = [
    "https://pubmed.ncbi.nlm.nih.gov/rss/search/1WnQjLbmkKziVwMd.../",  # CRISPR
    "https://pubmed.ncbi.nlm.nih.gov/rss/search/3k2398dkf.../",         # Gene therapy
]

fetcher = PubMedFetcher(rss_urls=rss_feeds)
articles = fetcher.fetch()

print(f"Fetched {len(articles)} articles.")
