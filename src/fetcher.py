import feedparser
import datetime
import json
from pathlib import Path

class PubMedFetcher:
    def __init__(self, rss_urls, storage_path="data/pubmed_raw.json"):
        self.rss_urls = rss_urls
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def fetch(self):
        """Fetch articles from all RSS URLs."""
        all_articles = []

        for url in self.rss_urls:
            feed = feedparser.parse(url)

            for entry in feed.entries:
                article = {
                    "title": entry.get("title"),
                    "summary": entry.get("summary", ""),
                    "link": entry.get("link"),
                    "published": entry.get("published"),
                    "source": "PubMed",
                    "fetched_at": datetime.datetime.utcnow().isoformat()
                }
                all_articles.append(article)

        self._save(all_articles)
        return all_articles

    def _save(self, articles):
        """Save raw fetched articles."""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2)

