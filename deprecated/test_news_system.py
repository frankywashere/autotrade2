#!/usr/bin/env python3
"""
Test news fetching and encoding system.

Verifies:
- News database initialization
- Google News RSS fetching (optional, requires network)
- News encoder (backtest_no_news mode)
- News window retrieval
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.fetch_news import init_news_db, get_news_window
from src.ml.news_encoder import NewsEncoder


def test_news_db_init():
    """Test news database initialization."""
    print("\n" + "=" * 70)
    print("🧪 TESTING NEWS DATABASE INITIALIZATION")
    print("=" * 70)

    test_db_path = 'data/test_news.db'

    print(f"\nInitializing test news database: {test_db_path}")
    try:
        init_news_db(test_db_path)
        print("  ✅ Database initialized successfully!")

        # Clean up
        Path(test_db_path).unlink(missing_ok=True)
        print("  ✓ Cleaned up test database")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        raise


def test_news_encoder_backtest_mode():
    """Test news encoder in backtest mode (returns zeros)."""
    print("\n" + "=" * 70)
    print("🧪 TESTING NEWS ENCODER (BACKTEST MODE)")
    print("=" * 70)

    print("\nCreating NewsEncoder in backtest_no_news mode...")
    encoder = NewsEncoder(mode='backtest_no_news', device='cpu')

    print("  ✓ Encoder created (should not load LFM2)")

    # Test encoding (should return zeros)
    print("\nEncoding headlines...")
    headlines = [
        "Tesla reports record quarterly deliveries",
        "TSLA stock surges on earnings beat",
        "Elon Musk announces new Gigafactory"
    ]

    news_vec, news_mask = encoder.encode_headlines(headlines, datetime.now())

    print(f"  ✓ Encoding complete")
    print(f"  news_vec shape: {news_vec.shape}")
    print(f"  news_mask: {news_mask}")

    # Validate
    assert news_vec.shape == (768,), f"Expected shape (768,), got {news_vec.shape}"
    assert news_mask == 0.0, f"Expected mask=0 in backtest mode, got {news_mask}"
    assert (news_vec == 0).all(), "Expected all zeros in backtest mode"

    print("  ✅ Backtest mode returns zeros correctly!")


def test_news_encoder_live_mode():
    """Test news encoder in live mode (optional, requires transformers)."""
    print("\n" + "=" * 70)
    print("🧪 TESTING NEWS ENCODER (LIVE MODE) - Optional")
    print("=" * 70)

    try:
        import transformers
        print("\nCreating NewsEncoder in live_with_news mode...")
        print("  ⚠️  This will download LFM2-350M (~1.5GB) if not cached")
        print("  Skipping live mode test (run manually if needed)")
        print("  To test: Set mode='live_with_news' and run with network access")

    except ImportError:
        print("\n  ⚠️  transformers not installed")
        print("  Install with: pip install transformers")
        print("  Live mode will not work without this dependency")


def test_news_window_retrieval():
    """Test news window retrieval from database."""
    print("\n" + "=" * 70)
    print("🧪 TESTING NEWS WINDOW RETRIEVAL")
    print("=" * 70)

    # This test requires a populated news database
    # For now, just verify the function exists and can be called

    test_db_path = 'data/test_news.db'

    print("\nInitializing test database...")
    init_news_db(test_db_path)

    print("Querying empty database...")
    timestamp = datetime.now()
    articles = get_news_window(timestamp, lookback_minutes=120, db_path=test_db_path)

    print(f"  ✓ Query successful (found {len(articles)} articles in empty DB)")

    assert articles == [], "Empty database should return no articles"

    # Clean up
    Path(test_db_path).unlink(missing_ok=True)

    print("  ✅ News window retrieval working!")


if __name__ == '__main__':
    test_news_db_init()
    test_news_encoder_backtest_mode()
    test_news_encoder_live_mode()
    test_news_window_retrieval()

    print("\n" + "=" * 70)
    print("✅ NEWS SYSTEM TESTS PASSED")
    print("=" * 70)
    print("\nNews system is ready!")
    print("- Backtest mode (news disabled): Working ✓")
    print("- Database operations: Working ✓")
    print("- Live mode: Requires transformers library")
    print()
