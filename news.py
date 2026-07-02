"""
News sentiment agent.

Pulls recent Google News RSS items for Indonesian market keywords and scores
headlines with VADER. This is a v1 placeholder — VADER is English-biased,
so Indonesian-language headlines are under-weighted. For production, swap
in XLM-RoBERTa or a FinBERT-multilingual model.

Note on compliance: we set a descriptive User-Agent and cache aggressively.
Headlines are used only to compute an aggregate sentiment score; individual
headlines are never reproduced verbatim in the app UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Identify ourselves — good citizenship, makes enforcement less likely
USER_AGENT = "IDXFearGreed/1.0 (educational/research; github.com/richardohugo/idx-fear-greed)"


QUERIES = [
    "Indonesia stock market",
    "IHSG Jakarta Composite",
    "Bank Indonesia rate",
    "Indonesia rupiah",
    "IDX equity",
]

RSS_TEMPLATE = "https://news.google.com/rss/search?q={q}&hl=en-ID&gl=ID&ceid=ID:en"


@dataclass
class Headline:
    title: str
    source: str
    published: datetime
    score: float  # -1..+1 VADER compound


def fetch_headlines(max_age_hours: int = 48, limit_per_query: int = 15) -> list[Headline]:
    analyzer = SentimentIntensityAnalyzer()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    seen_titles: set[str] = set()
    out: list[Headline] = []

    for q in QUERIES:
        url = RSS_TEMPLATE.format(q=q.replace(" ", "+"))
        feed = feedparser.parse(url, agent=USER_AGENT)
        for entry in feed.entries[:limit_per_query]:
            title = entry.get("title", "").strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)

            # Parse publish date
            try:
                pub = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            except (AttributeError, TypeError):
                pub = datetime.now(timezone.utc)

            if pub < cutoff:
                continue

            source = entry.get("source", {}).get("title", "Unknown")
            compound = analyzer.polarity_scores(title)["compound"]
            out.append(Headline(title=title, source=source, published=pub, score=compound))

    out.sort(key=lambda h: h.published, reverse=True)
    return out


def aggregate_sentiment(headlines: list[Headline]) -> tuple[float, int]:
    """Return a 0-100 greed score and the headline count."""
    if not headlines:
        return 50.0, 0
    mean_compound = sum(h.score for h in headlines) / len(headlines)
    # VADER compound is -1..+1; map to 0..100
    score = (mean_compound + 1) / 2 * 100
    return float(score), len(headlines)
