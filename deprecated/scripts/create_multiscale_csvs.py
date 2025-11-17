#!/usr/bin/env python3
"""
Generate multi-timeframe CSV files from 1-minute data.

This script resamples TSLA_1min.csv and SPY_1min.csv to create
CSV files for all training timeframes:
- 5min, 15min, 30min
- 1hour, 2hour, 3hour, 4hour
- daily, weekly, monthly, 3month

Usage:
    python scripts/create_multiscale_csvs.py
"""

import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import DATA_DIR


# Timeframe definitions (name → pandas resample rule)
TIMEFRAMES = {
    '5min': '5T',
    '15min': '15T',
    '30min': '30T',
    '1hour': '1h',
    '2hour': '2h',
    '3hour': '3h',
    '4hour': '4h',
    'daily': '1D',
    'weekly': '1W',
    'monthly': '1M',
    '3month': '3M'
}


def load_1min_data(symbol):
    """Load 1-minute CSV data"""
    csv_path = DATA_DIR / f"{symbol}_1min.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"1-minute data not found: {csv_path}")

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')

    print(f"  ✓ Loaded {len(df):,} bars ({df.index[0]} to {df.index[-1]})")

    return df


def resample_to_timeframe(df, timeframe_name, resample_rule):
    """Resample 1-minute data to specified timeframe"""
    print(f"\nResampling to {timeframe_name}...")

    # Resample OHLCV data
    resampled = df.resample(resample_rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    print(f"  ✓ Created {len(resampled):,} {timeframe_name} bars")
    print(f"    Range: {resampled.index[0]} to {resampled.index[-1]}")

    return resampled


def save_csv(df, symbol, timeframe_name):
    """Save resampled data to CSV"""
    output_path = DATA_DIR / f"{symbol}_{timeframe_name}.csv"

    df.to_csv(output_path)
    print(f"  ✓ Saved to {output_path}")

    # Calculate file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"    Size: {size_mb:.2f} MB")


def main():
    """Generate all multi-scale CSV files"""
    print("=" * 70)
    print("MULTI-SCALE CSV GENERATION")
    print("=" * 70)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Generating {len(TIMEFRAMES)} timeframes for TSLA and SPY")
    print()

    # Process each symbol
    for symbol in ['TSLA', 'SPY']:
        print("\n" + "=" * 70)
        print(f"Processing {symbol}")
        print("=" * 70)

        # Load 1-minute data
        try:
            df_1min = load_1min_data(symbol)
        except FileNotFoundError as e:
            print(f"  ✗ Error: {e}")
            print(f"  Skipping {symbol}")
            continue

        # Create resampled versions
        for tf_name, tf_rule in TIMEFRAMES.items():
            try:
                resampled_df = resample_to_timeframe(df_1min, tf_name, tf_rule)
                save_csv(resampled_df, symbol, tf_name)
            except Exception as e:
                print(f"  ✗ Error resampling {tf_name}: {e}")
                continue

    print("\n" + "=" * 70)
    print("✅ MULTI-SCALE CSV GENERATION COMPLETE")
    print("=" * 70)

    # Summary
    print("\nGenerated files:")
    csv_files = sorted(DATA_DIR.glob("*_[0-9]*min.csv")) + \
                sorted(DATA_DIR.glob("*_[0-9]*hour.csv")) + \
                sorted(DATA_DIR.glob("*_daily.csv")) + \
                sorted(DATA_DIR.glob("*_weekly.csv")) + \
                sorted(DATA_DIR.glob("*_monthly.csv")) + \
                sorted(DATA_DIR.glob("*_3month.csv"))

    total_size = 0
    for csv_file in csv_files:
        size_mb = csv_file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  {csv_file.name:<30} {size_mb:>8.2f} MB")

    print(f"\nTotal size: {total_size:.2f} MB")
    print(f"Total files: {len(csv_files)}")

    print("\n" + "=" * 70)
    print("Ready for multi-scale training!")
    print("=" * 70)
    print("\nExample usage:")
    print("  python train_model_lazy.py --input_timeframe 15min --sequence_length 500")
    print("  python train_model_lazy.py --input_timeframe 1hour --sequence_length 500")
    print("  python train_model_lazy.py --input_timeframe 4hour --sequence_length 500")
    print("  python train_model_lazy.py --input_timeframe daily --sequence_length 500")
    print()


if __name__ == '__main__':
    main()
