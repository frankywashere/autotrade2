#!/usr/bin/env python3
"""Test script to verify continuation labels fix."""

import pandas as pd
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.features import TradingFeatureExtractor
from src.ml.data_feed import CSVDataFeed

def main():
    print("Testing continuation labels fix...")

    # Initialize feature extractor
    extractor = TradingFeatureExtractor()

    # Load data
    data_feed = CSVDataFeed(data_dir="datasets/1min_raw", timeframe="1min_raw")

    # Get a small sample of data - load TSLA data
    df = data_feed.load_data("TSLA",
                            start_date="2017-01-03",
                            end_date="2017-01-10")

    if df.empty:
        print("No data loaded")
        return

    # Prefix columns with tsla_ as expected by feature extractor
    df = df.rename(columns={col: f'tsla_{col}' for col in df.columns})

    print(f"Loaded {len(df)} rows of data")
    print(f"Columns: {df.columns.tolist()}")

    # Try to generate continuation labels with debug output
    print("\nGenerating continuation labels...")
    timestamps = df.index[:100].tolist()  # Just test first 100 timestamps

    try:
        labels_df = extractor.generate_continuation_labels(
            df,
            timestamps,
            prediction_horizon=24,
            debug=True
        )
        print(f"\n✓ Successfully generated {len(labels_df)} continuation labels!")
        if not labels_df.empty:
            print(f"Label columns: {labels_df.columns.tolist()}")
            print(f"Sample labels:\n{labels_df.head()}")
    except Exception as e:
        print(f"\n✗ Error generating labels: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()