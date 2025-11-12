"""
News fetching module for market and TSLA-specific news.

Fetches news via:
1. Google News RSS (no API key needed)
2. Finnhub API (optional, if API key provided)

Stores headlines in news.db with timestamps for leak-safe retrieval.
Filters by whitelisted sources and excludes promotional content.
"""

import feedparser
import sqlite3
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import time


# Query strings for news fetching
TSLA_QUERY = (
    "Tesla OR TSLA OR 'Tesla, Inc.' OR 'Elon Musk' OR Cybertruck OR "
    "'Giga Texas' OR 'Giga Berlin' OR 'Giga Shanghai' OR "
    "'Full Self-Driving' OR FSD OR Autopilot OR NHTSA OR recall OR "
    "Supercharger OR 4680"
)

MARKET_QUERY = (
    "FOMC OR 'Federal Reserve' OR 'Fed minutes' OR CPI OR PPI OR "
    "'Core PCE' OR 'jobs report' OR NFP OR ISM OR PMI OR "
    "'Treasury yields' OR '10-year yield' OR 'rate hike' OR 'rate cut' OR "
    "'quantitative tightening' OR 'oil prices' OR OPEC OR "
    "'Middle East tensions' OR 'government shutdown' OR 'debt ceiling' OR "
    "'S&P 500' OR Nasdaq OR VIX OR GDP"
)

# Whitelisted news sources (high-quality only)
WHITELIST_SOURCES = [
    'reuters.com',
    'apnews.com',
    'ap.org',
    'bloomberg.com',
    'wsj.com',
    'ft.com',
    'cnbc.com',
    'marketwatch.com',
    'barrons.com',
    'finance.yahoo.com',
    'investing.com',
    'seekingalpha.com'
]

# Exclude promotional/low-quality content
EXCLUDE_PATTERN = re.compile(
    r'(?i)review|coupon|deal|giveaway|celebrity|rumor|'
    r'sponsored|advertisement|promo|discount|sale',
    re.IGNORECASE
)


def init_news_db(db_path: str = 'data/news.db'):
    """
    Initialize news database schema.

    Args:
        db_path: Path to SQLite database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            title TEXT NOT NULL,
            source TEXT,
            url TEXT,
            query_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(timestamp, title)
        )
    """)

    # Create index for fast timestamp queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_news_timestamp
        ON news(timestamp)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_news_query_type
        ON news(query_type)
    """)

    conn.commit()
    conn.close()


def fetch_google_news_rss(query: str, max_results: int = 20) -> List[Dict]:
    """
    Fetch news from Google News RSS feed.

    No API key required, but rate-limited.

    Args:
        query: Search query string
        max_results: Maximum number of articles to fetch

    Returns:
        List of article dictionaries
    """
    # Google News RSS URL
    base_url = "https://news.google.com/rss/search"
    params = f"?q={query}&hl=en-US&gl=US&ceid=US:en"
    url = base_url + params

    try:
        feed = feedparser.parse(url)
    except Exception as e:
        print(f"Error fetching Google News RSS: {e}")
        return []

    articles = []

    for entry in feed.entries[:max_results]:
        try:
            # Parse entry
            title = entry.get('title', '')
            link = entry.get('link', '')
            published = entry.get('published', '')

            # Extract source from title (Google News format: "Title - Source")
            source = ''
            if ' - ' in title:
                title_parts = title.rsplit(' - ', 1)
                title = title_parts[0]
                source = title_parts[1] if len(title_parts) > 1 else ''

            # Filter by whitelist
            if not any(s in link.lower() or s in source.lower() for s in WHITELIST_SOURCES):
                continue

            # Exclude promotional content
            if EXCLUDE_PATTERN.search(title):
                continue

            # Parse timestamp
            try:
                # Google RSS uses RFC 2822 format
                timestamp = datetime.strptime(published, '%a, %d %b %Y %H:%M:%S %Z')
            except:
                # Fallback to current time
                timestamp = datetime.now()

            articles.append({
                'title': title,
                'source': source,
                'url': link,
                'timestamp': timestamp.isoformat(),
            })

        except Exception as e:
            print(f"Error parsing entry: {e}")
            continue

    return articles


def fetch_finnhub_news(api_key: str, symbol: str = 'TSLA', max_results: int = 20) -> List[Dict]:
    """
    Fetch news from Finnhub API (requires API key).

    Args:
        api_key: Finnhub API key
        symbol: Stock symbol
        max_results: Maximum articles

    Returns:
        List of article dictionaries
    """
    try:
        import requests
    except ImportError:
        print("requests library required for Finnhub. Install with: pip install requests")
        return []

    # Finnhub company news endpoint
    url = f"https://finnhub.io/api/v1/company-news"

    # Get news from last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    params = {
        'symbol': symbol,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'token': api_key
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error fetching Finnhub news: {e}")
        return []

    articles = []

    for item in data[:max_results]:
        try:
            title = item.get('headline', '')
            source = item.get('source', '')
            url_link = item.get('url', '')
            timestamp_unix = item.get('datetime', 0)

            # Convert Unix timestamp to datetime
            timestamp = datetime.fromtimestamp(timestamp_unix)

            # Filter by whitelist
            if source.lower() not in [s.replace('.com', '') for s in WHITELIST_SOURCES]:
                continue

            # Exclude promotional content
            if EXCLUDE_PATTERN.search(title):
                continue

            articles.append({
                'title': title,
                'source': source,
                'url': url_link,
                'timestamp': timestamp.isoformat(),
            })

        except Exception as e:
            print(f"Error parsing Finnhub article: {e}")
            continue

    return articles


def store_news_to_db(articles: List[Dict], query_type: str, db_path: str = 'data/news.db'):
    """
    Store news articles to database.

    Args:
        articles: List of article dictionaries
        query_type: 'TSLA' or 'MARKET'
        db_path: Path to SQLite database
    """
    if not articles:
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    inserted = 0
    duplicates = 0

    for article in articles:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO news (timestamp, title, source, url, query_type, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                article['timestamp'],
                article['title'],
                article.get('source', ''),
                article.get('url', ''),
                query_type,
                datetime.now().isoformat()
            ))

            if cursor.rowcount > 0:
                inserted += 1
            else:
                duplicates += 1

        except Exception as e:
            print(f"Error inserting article: {e}")
            continue

    conn.commit()
    conn.close()

    print(f"  ✓ Inserted {inserted} new articles ({duplicates} duplicates skipped)")


def get_news_window(timestamp: datetime,
                    lookback_minutes: int = 120,
                    query_types: Optional[List[str]] = None,
                    db_path: str = 'data/news.db') -> List[Dict]:
    """
    Retrieve news from time window [T - lookback, T].

    Ensures leak-safe retrieval: only news published BEFORE timestamp.

    Args:
        timestamp: Current prediction time
        lookback_minutes: Minutes to look back (default: 120 = 2 hours)
        query_types: List of query types to filter (e.g., ['TSLA', 'MARKET'])
        db_path: Path to news database

    Returns:
        List of news articles as dictionaries
    """
    conn = sqlite3.connect(db_path)

    # Calculate time window
    end_time = timestamp
    start_time = timestamp - timedelta(minutes=lookback_minutes)

    # Build query
    query = """
        SELECT id, timestamp, title, source, url, query_type
        FROM news
        WHERE timestamp >= ? AND timestamp < ?
    """
    params = [start_time.isoformat(), end_time.isoformat()]

    if query_types:
        placeholders = ','.join(['?'] * len(query_types))
        query += f" AND query_type IN ({placeholders})"
        params.extend(query_types)

    query += " ORDER BY timestamp DESC"

    cursor = conn.cursor()
    cursor.execute(query, params)

    articles = []
    for row in cursor.fetchall():
        articles.append({
            'id': row[0],
            'timestamp': row[1],
            'title': row[2],
            'source': row[3],
            'url': row[4],
            'query_type': row[5]
        })

    conn.close()

    return articles


def fetch_and_store_news(tsla_max: int = 10,
                         market_max: int = 10,
                         finnhub_api_key: Optional[str] = None,
                         db_path: str = 'data/news.db'):
    """
    Fetch news from all sources and store to database.

    Args:
        tsla_max: Maximum TSLA articles to fetch
        market_max: Maximum MARKET articles to fetch
        finnhub_api_key: Optional Finnhub API key
        db_path: Path to news database
    """
    print("=" * 70)
    print("FETCHING NEWS")
    print("=" * 70)

    # Initialize database
    init_news_db(db_path)

    # Fetch TSLA news
    print("\nFetching TSLA news from Google News RSS...")
    tsla_articles = fetch_google_news_rss(TSLA_QUERY, max_results=tsla_max)
    print(f"  Found {len(tsla_articles)} TSLA articles")
    store_news_to_db(tsla_articles, 'TSLA', db_path)

    # Fetch MARKET news
    print("\nFetching MARKET news from Google News RSS...")
    market_articles = fetch_google_news_rss(MARKET_QUERY, max_results=market_max)
    print(f"  Found {len(market_articles)} MARKET articles")
    store_news_to_db(market_articles, 'MARKET', db_path)

    # Optional: Fetch from Finnhub if API key provided
    if finnhub_api_key:
        print("\nFetching TSLA news from Finnhub API...")
        finnhub_articles = fetch_finnhub_news(finnhub_api_key, symbol='TSLA', max_results=tsla_max)
        print(f"  Found {len(finnhub_articles)} TSLA articles")
        store_news_to_db(finnhub_articles, 'TSLA', db_path)

    print("\n" + "=" * 70)
    print("✅ NEWS FETCH COMPLETE")
    print("=" * 70)

    # Summary
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM news WHERE query_type = 'TSLA'")
    tsla_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM news WHERE query_type = 'MARKET'")
    market_count = cursor.fetchone()[0]

    cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM news")
    date_range = cursor.fetchone()

    conn.close()

    print(f"\nDatabase summary:")
    print(f"  TSLA articles: {tsla_count}")
    print(f"  MARKET articles: {market_count}")
    print(f"  Date range: {date_range[0]} to {date_range[1]}")


# CLI interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fetch news for trading predictions')
    parser.add_argument('--tsla_max', type=int, default=10,
                       help='Maximum TSLA articles to fetch')
    parser.add_argument('--market_max', type=int, default=10,
                       help='Maximum MARKET articles to fetch')
    parser.add_argument('--finnhub_key', type=str, default=None,
                       help='Finnhub API key (optional)')
    parser.add_argument('--db_path', type=str, default='data/news.db',
                       help='Path to news database')

    args = parser.parse_args()

    fetch_and_store_news(
        tsla_max=args.tsla_max,
        market_max=args.market_max,
        finnhub_api_key=args.finnhub_key,
        db_path=args.db_path
    )


__all__ = [
    'init_news_db',
    'fetch_google_news_rss',
    'fetch_finnhub_news',
    'store_news_to_db',
    'get_news_window',
    'fetch_and_store_news'
]
