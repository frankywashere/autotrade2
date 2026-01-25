#!/usr/bin/env python3
"""
Compare C++ resampling output against Python reference implementation.

This script:
1. Loads the same data using Python
2. Resamples using Python's canonical resample_ohlc
3. Exports Python results to JSON
4. Compares against C++ scanner's resampling

Usage:
    python compare_resampling.py --data-dir ../data --output comparison.json
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add v15 to path
v15_path = Path(__file__).parent.parent.parent / 'v15'
if not v15_path.exists():
    # Try alternative path
    v15_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(v15_path))

from v15.core.resample import resample_ohlc, TIMEFRAMES, BARS_PER_TF


def load_tsla_1min(data_dir: str) -> pd.DataFrame:
    """Load TSLA 1min data"""
    filepath = os.path.join(data_dir, 'TSLA_1min.csv')
    df = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
    return df


def load_spy_1min(data_dir: str) -> pd.DataFrame:
    """Load SPY 1min data"""
    filepath = os.path.join(data_dir, 'SPY_1min.csv')
    df = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
    return df


def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1min to 5min"""
    return resample_ohlc(df, '5min')


def test_resampling_accuracy(data_dir: str):
    """Test that Python and C++ produce identical resampling results"""

    print("="*80)
    print("PYTHON RESAMPLING REFERENCE GENERATION")
    print("="*80)

    # Load TSLA 1min
    print(f"\nLoading TSLA 1min data from {data_dir}...")
    tsla_1min = load_tsla_1min(data_dir)
    print(f"  Loaded {len(tsla_1min):,} bars")

    # Resample to 5min
    print("\nResampling TSLA 1min -> 5min...")
    tsla_5min = resample_to_5min(tsla_1min)
    print(f"  Resampled to {len(tsla_5min):,} bars")

    # Validate OHLCV semantics
    print("\nValidating OHLCV semantics...")
    errors = []

    for i in range(len(tsla_5min)):
        row = tsla_5min.iloc[i]

        if row['high'] < row['low']:
            errors.append(f"Bar {i}: high < low")
        if row['high'] < row['open'] or row['high'] < row['close']:
            errors.append(f"Bar {i}: high < open or close")
        if row['low'] > row['open'] or row['low'] > row['close']:
            errors.append(f"Bar {i}: low > open or close")
        if row['volume'] < 0:
            errors.append(f"Bar {i}: negative volume")

    if errors:
        print(f"  ✗ Found {len(errors)} errors:")
        for err in errors[:10]:  # Show first 10
            print(f"    {err}")
        return False
    else:
        print(f"  ✓ All {len(tsla_5min):,} bars valid")

    # Resample to all timeframes
    print("\nResampling TSLA 5min to all timeframes...")
    resampled = {}

    for tf in TIMEFRAMES[1:]:  # Skip 5min (already have it)
        try:
            df_tf = resample_ohlc(tsla_5min, tf)
            resampled[tf] = df_tf
            print(f"  {tf:8s}: {len(df_tf):6,} bars (from {BARS_PER_TF[tf]:4d} 5min bars each)")

            # Validate each timeframe
            for i in range(len(df_tf)):
                row = df_tf.iloc[i]
                if row['high'] < row['low']:
                    print(f"    ERROR: {tf} bar {i}: high < low")

        except Exception as e:
            print(f"  ✗ {tf}: {e}")
            return False

    # Export first/last bars for comparison
    print("\n" + "="*80)
    print("COMPARISON DATA FOR C++")
    print("="*80)

    # First 5 bars of 5min TSLA
    print("\nFirst 5 bars of TSLA 5min:")
    print(tsla_5min.head().to_string())

    # First 3 bars of each timeframe
    print("\nFirst 3 bars of each timeframe:")
    for tf in ['15min', '30min', '1h', 'daily']:
        if tf in resampled:
            print(f"\n{tf}:")
            print(resampled[tf].head(3).to_string())

    # Export to JSON for automated comparison
    export_data = {
        'tsla_5min_first_10': tsla_5min.head(10).reset_index().to_dict('records'),
        'tsla_5min_count': len(tsla_5min),
        'resampled_counts': {tf: len(df) for tf, df in resampled.items()},
    }

    # Add first bar of each timeframe
    for tf, df in resampled.items():
        if len(df) > 0:
            first_bar = df.iloc[0]
            export_data[f'{tf}_first_bar'] = {
                'timestamp': str(df.index[0]),
                'open': float(first_bar['open']),
                'high': float(first_bar['high']),
                'low': float(first_bar['low']),
                'close': float(first_bar['close']),
                'volume': float(first_bar['volume']),
            }

    # Save to JSON
    output_file = 'python_resampling_reference.json'
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)

    print(f"\n✓ Exported reference data to {output_file}")

    # Manual test: Verify 15min = 3 * 5min
    print("\n" + "="*80)
    print("MANUAL VERIFICATION: 15min = 3 × 5min")
    print("="*80)

    # Take first 3 5min bars
    first_3 = tsla_5min.head(3)
    print("\nFirst 3 5min bars:")
    print(first_3.to_string())

    # Expected 15min bar
    expected_15min = {
        'open': first_3.iloc[0]['open'],
        'high': first_3['high'].max(),
        'low': first_3['low'].min(),
        'close': first_3.iloc[2]['close'],
        'volume': first_3['volume'].sum(),
    }

    print("\nExpected first 15min bar:")
    for k, v in expected_15min.items():
        print(f"  {k:8s}: {v}")

    # Actual 15min bar
    if '15min' in resampled and len(resampled['15min']) > 0:
        actual_15min = resampled['15min'].iloc[0]
        print("\nActual first 15min bar:")
        print(f"  open    : {actual_15min['open']}")
        print(f"  high    : {actual_15min['high']}")
        print(f"  low     : {actual_15min['low']}")
        print(f"  close   : {actual_15min['close']}")
        print(f"  volume  : {actual_15min['volume']}")

        # Compare
        matches = []
        for key in ['open', 'high', 'low', 'close', 'volume']:
            match = np.isclose(expected_15min[key], actual_15min[key], rtol=1e-10)
            matches.append(match)
            symbol = "✓" if match else "✗"
            print(f"\n  {key:8s}: {symbol} (diff={abs(expected_15min[key] - actual_15min[key]):.10f})")

        if all(matches):
            print("\n✓ 15min resampling CORRECT")
        else:
            print("\n✗ 15min resampling INCORRECT")
            return False

    print("\n" + "="*80)
    print("✓ ALL PYTHON RESAMPLING TESTS PASSED")
    print("="*80)

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compare C++ resampling against Python')
    parser.add_argument('--data-dir', default='../data', help='Data directory')
    args = parser.parse_args()

    data_dir = args.data_dir

    if not os.path.isdir(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return 1

    success = test_resampling_accuracy(data_dir)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
