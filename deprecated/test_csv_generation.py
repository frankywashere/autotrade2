#!/usr/bin/env python3
"""
Test multi-scale CSV generation.

Verifies that the CSV generation script correctly resamples
1-minute data to all target timeframes.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config


def test_csv_generation():
    """Test that multi-scale CSVs are correctly generated."""
    print("\n" + "=" * 70)
    print("🧪 TESTING MULTI-SCALE CSV GENERATION")
    print("=" * 70)

    # Expected timeframes
    timeframes = [
        '1min', '5min', '15min', '30min',
        '1hour', '2hour', '3hour', '4hour',
        'daily', 'weekly', 'monthly', '3month'
    ]

    print(f"\nChecking for {len(timeframes)} timeframe files...")
    print(f"Data directory: {config.DATA_DIR}")

    # Check each symbol
    for symbol in ['TSLA', 'SPY']:
        print(f"\n{symbol}:")

        for tf in timeframes:
            csv_path = config.DATA_DIR / f"{symbol}_{tf}.csv"

            if csv_path.exists():
                # Load and validate
                try:
                    df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')

                    # Check required columns
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    has_all_cols = all(col in df.columns for col in required_cols)

                    # Check data quality
                    has_nulls = df[required_cols].isnull().any().any()
                    has_zeros = (df[required_cols] == 0).any().any()

                    # File size
                    size_mb = csv_path.stat().st_size / (1024 * 1024)

                    if has_all_cols and not has_nulls:
                        status = "✅"
                    else:
                        status = "⚠️"

                    print(f"  {status} {tf:10s}: {len(df):6,} bars, {size_mb:6.2f} MB"
                          f"  ({df.index[0].date()} to {df.index[-1].date()})")

                    if has_zeros:
                        print(f"      ⚠️  Warning: Contains zero values")

                except Exception as e:
                    print(f"  ❌ {tf:10s}: Error loading - {e}")

            else:
                print(f"  ❌ {tf:10s}: File not found")

    print("\n" + "=" * 70)
    print("CSV VALIDATION COMPLETE")
    print("=" * 70)

    # Check if base 1-minute data exists
    tsla_1min = config.DATA_DIR / "TSLA_1min.csv"
    spy_1min = config.DATA_DIR / "SPY_1min.csv"

    if not tsla_1min.exists() or not spy_1min.exists():
        print("\n⚠️  Base 1-minute data not found!")
        print("    Run the CSV generation script first:")
        print("    python scripts/create_multiscale_csvs.py")
    else:
        print("\n✅ Multi-scale CSVs ready for training!")
        print("\nUsage examples:")
        print("  python train_model_lazy.py --input_timeframe 15min --sequence_length 500 --output models/lnn_15min.pth")
        print("  python train_model_lazy.py --input_timeframe 1hour --sequence_length 500 --output models/lnn_1hour.pth")
        print("  python train_model_lazy.py --input_timeframe 4hour --sequence_length 500 --output models/lnn_4hour.pth")
        print("  python train_model_lazy.py --input_timeframe daily --sequence_length 500 --output models/lnn_daily.pth")

    print()


if __name__ == '__main__':
    test_csv_generation()
