#!/usr/bin/env python3
"""Test data loading with actual data files."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.data_feed import CSVDataFeed
import config

def main():
    print("Testing data loading...")
    print(f"Data directory: {config.DATA_DIR}")

    # Check if data files exist
    tsla_path = config.DATA_DIR / "TSLA_1min.csv"
    spy_path = config.DATA_DIR / "SPY_1min.csv"

    print(f"\nChecking files:")
    print(f"  TSLA_1min.csv exists: {tsla_path.exists()}")
    print(f"  SPY_1min.csv exists: {spy_path.exists()}")

    # Try to load data
    data_feed = CSVDataFeed(timeframe="1min")

    print("\nLoading TSLA data...")
    try:
        tsla_df = data_feed.load_data("TSLA", start_date="2017-01-03", end_date="2017-01-10")
        print(f"  ✓ Loaded {len(tsla_df)} rows")
        print(f"  Columns: {tsla_df.columns.tolist()}")
        print(f"  Date range: {tsla_df.index.min()} to {tsla_df.index.max()}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("\nLoading SPY data...")
    try:
        spy_df = data_feed.load_data("SPY", start_date="2017-01-03", end_date="2017-01-10")
        print(f"  ✓ Loaded {len(spy_df)} rows")
        print(f"  Columns: {spy_df.columns.tolist()}")
        print(f"  Date range: {spy_df.index.min()} to {spy_df.index.max()}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

if __name__ == "__main__":
    main()