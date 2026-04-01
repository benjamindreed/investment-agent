"""
News fetching and sentiment analysis.

Sources (in priority order):
  1. NewsAPI     (requires NEWS_API_KEY — 100 req/day on free tier)
  2. X.com posts (requires X_BEARER_TOKEN — Basic tier $100/mo or above)
  3. yfinance built-in news feed
  4. Google News RSS (no key required)

Sentiment is scored with VADER; a stock is flagged as having "negative media
coverage" if its compound score average across recent articles is < -0.05.
"""
import os
import time
import requests
import feedparser
import yfinance as yf
from datetime import datetime, timedelta, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

NEGATIVE_THRESHOLD = -0.05  # VADER compound score cutoff


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score(text: str) -> float:
    return _analyzer.polarity_scores(text)["compound"]


def _newsapi_articles(ticker: str, company_name: str, api_key: str, days: int = 90) -> list[dict]:
    """Fetch articles from NewsAPI for a ticker/company."""
    from_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    query = f'"{company_name}" OR "{ticker}"'
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 30,
        "apiKey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "published_at": a.get("publishedAt", ""),
                "source": a.get("source", {}).get("name", "NewsAPI"),
                "url": a.get("url", ""),
            }
            for a in data.get("articles", [])
        ]
    except Exception:
        return []


def _yfinance_articles(ticker: str) -> list[dict]:
    """Fetch news from yfinance (typically last ~7-30 days)."""
    try:
        news = yf.Ticker(ticker).news or []
        results = []
        for item in news:
            content = item.get("content", {})
            title = content.get("title", "") if isinstance(content, dict) else ""
            summary = content.get("summary", "") if isinstance(content, dict) else ""
            # Support both old and new yfinance news schemas
            if not title:
                title = item.get("title", "")
            if not summary:
                summary = item.get("summary", "")

            pub_ts = None
            pub_raw = (
                content.get("pubDate") if isinstance(content, dict) else None
            ) or item.get("providerPublishTime")
            if isinstance(pub_raw, (int, float)):
                pub_ts = datetime.fromtimestamp(pub_raw, tz=timezone.utc).isoformat()
            elif isinstance(pub_raw, str):
                pub_ts = pub_raw

            results.append(
                {
                    "title": title,
                    "description": summary,
                    "published_at": pub_ts or "",
                    "source": "Yahoo Finance",
                    "url": item.get("link", "") or (
                        content.get("canonicalUrl", {}).get("url", "")
                        if isinstance(content, dict) else ""
                    ),
                }
            )
        return results
    except Exception:
        return []


def _x_posts(ticker: str, company_name: str, bearer_token: str, days: int = 35) -> list[dict]:
    """
    Fetch recent posts from X.com (Twitter) via the v2 search endpoint.
    Requires a Bearer Token from developer.x.com (Basic tier or above).
    Returns an empty list silently if the token is missing or the request fails.
    """
    if not bearer_token:
        return []
    start_time = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    # Exclude retweets; target cashtag and company name
    query = f'(${ticker} OR "{company_name}") -is:retweet lang:en'
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    params = {
        "query": query,
        "start_time": start_time,
        "max_results": 100,
        "tweet.fields": "created_at,text",
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for tweet in data.get("data", []):
            results.append(
                {
                    "title": tweet.get("text", "")[:200],
                    "description": "",
                    "published_at": tweet.get("created_at", ""),
                    "source": "X.com",
                    "url": f"https://x.com/i/web/status/{tweet.get('id', '')}",
                }
            )
        return results
    except Exception:
        return []


def _gnews_rss_articles(ticker: str, company_name: str) -> list[dict]:
    """Scrape Google News RSS (no API key required)."""
    query = f"{company_name} stock".replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        feed = feedparser.parse(url)
        cutoff = datetime.now(timezone.utc) - timedelta(days=90)
        results = []
        for entry in feed.entries[:30]:
            pub = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            if pub and pub < cutoff:
                continue
            results.append(
                {
                    "title": entry.get("title", ""),
                    "description": entry.get("summary", ""),
                    "published_at": pub.isoformat() if pub else "",
                    "source": "Google News",
                    "url": entry.get("link", ""),
                }
            )
        return results
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def analyze_sentiment(
    ticker: str,
    company_name: str,
    news_api_key: str = "",
    days: int = 90,
    x_bearer_token: str = "",
) -> dict:
    """
    Collect news for a ticker and compute aggregate VADER sentiment.

    Returns:
        {
            "ticker": str,
            "has_negative_coverage": bool,
            "avg_compound": float,
            "article_count": int,
            "negative_count": int,
            "articles": list[dict],  # each has title, source, compound, published_at, url
        }
    """
    articles_raw: list[dict] = []

    if news_api_key:
        articles_raw.extend(_newsapi_articles(ticker, company_name, news_api_key, days))

    if x_bearer_token:
        articles_raw.extend(_x_posts(ticker, company_name, x_bearer_token, days))

    # Always supplement with yfinance + Google News
    articles_raw.extend(_yfinance_articles(ticker))
    articles_raw.extend(_gnews_rss_articles(ticker, company_name))

    # Deduplicate by title
    seen_titles: set[str] = set()
    unique_articles = []
    for a in articles_raw:
        title_key = a["title"].strip().lower()[:80]
        if title_key and title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_articles.append(a)

    # Score each article
    scored = []
    for a in unique_articles:
        text = f"{a['title']} {a['description']}".strip()
        compound = _score(text)
        scored.append(
            {
                "title": a["title"],
                "source": a["source"],
                "published_at": a["published_at"],
                "url": a["url"],
                "compound": round(compound, 4),
            }
        )

    if not scored:
        return {
            "ticker": ticker,
            "has_negative_coverage": False,
            "avg_compound": 0.0,
            "article_count": 0,
            "negative_count": 0,
            "articles": [],
        }

    compounds = [a["compound"] for a in scored]
    avg = round(sum(compounds) / len(compounds), 4)
    neg_count = sum(1 for c in compounds if c < NEGATIVE_THRESHOLD)

    return {
        "ticker": ticker,
        "has_negative_coverage": avg < NEGATIVE_THRESHOLD,
        "avg_compound": avg,
        "article_count": len(scored),
        "negative_count": neg_count,
        "articles": sorted(scored, key=lambda x: x["compound"]),  # most negative first
    }


def filter_negative_coverage(
    tickers_info: list[dict],
    news_api_key: str = "",
    progress_callback=None,
) -> tuple[list[dict], dict[str, dict]]:
    """
    Given a list of {"ticker": ..., "name": ...} dicts, return only those with
    negative media coverage, plus the full sentiment data map.
    """
    sentiment_map: dict[str, dict] = {}
    passing: list[dict] = []
    total = len(tickers_info)

    for i, item in enumerate(tickers_info):
        ticker = item["ticker"]
        name = item.get("name", ticker)

        if progress_callback:
            progress_callback(i, total, f"Analysing news sentiment for {ticker} ({i+1}/{total})...")

        result = analyze_sentiment(ticker, name, news_api_key)
        sentiment_map[ticker] = result
        time.sleep(0.2)  # gentle rate limiting for Google News RSS

        if result["has_negative_coverage"]:
            passing.append(item)

    return passing, sentiment_map
