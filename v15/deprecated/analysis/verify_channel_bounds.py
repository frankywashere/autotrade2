#!/usr/bin/env python3
"""
Verify that channel bounds being drawn match the channel features.

For sample 109, window 10, SPY 5min:
1. Get the channel features: slope, intercept, std_dev_ratio
2. Reconstruct what the upper/lower bounds SHOULD be
3. Compare with what's being calculated in the inspector
4. Check if the blue lines on the chart match the calculated bounds
"""

import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path


def reconstruct_channel_bounds(slope, intercept, std_dev_ratio, avg_price, window):
    """
    Reconstruct channel bounds from features.

    Args:
        slope: Channel slope from features
        intercept: Channel intercept from features
        std_dev_ratio: Standard deviation ratio from features
        avg_price: Average price for computing actual std_dev
        window: Window size

    Returns:
        Tuple of (x_values, upper_line, lower_line, center_line)
    """
    # Compute actual std_dev from ratio
    std_dev = std_dev_ratio * avg_price

    # Create x-axis positions
    x = np.arange(window)

    # Reconstruct channel lines
    center_line = slope * x + intercept
    upper_line = center_line + 2.0 * std_dev
    lower_line = center_line - 2.0 * std_dev

    return x, upper_line, lower_line, center_line


def verify_sample(cache_path, data_dir, sample_idx=109, window=10, asset='spy', tf='5min'):
    """
    Verify channel bounds for a specific sample.

    Args:
        cache_path: Path to samples pickle file
        data_dir: Path to data directory
        sample_idx: Sample index to check
        window: Window size to check
        asset: Asset to check ('tsla' or 'spy')
        tf: Timeframe to check
    """
    print(f"\n{'='*80}")
    print(f"CHANNEL BOUNDS VERIFICATION")
    print(f"{'='*80}")
    print(f"Sample: {sample_idx}")
    print(f"Window: {window}")
    print(f"Asset: {asset.upper()}")
    print(f"Timeframe: {tf}")
    print(f"{'='*80}\n")

    # Load samples
    print(f"Loading samples from: {cache_path}")
    with open(cache_path, 'rb') as f:
        samples = pickle.load(f)
    print(f"  Loaded {len(samples)} samples\n")

    if sample_idx >= len(samples):
        print(f"ERROR: Sample index {sample_idx} out of range (max: {len(samples)-1})")
        return

    sample = samples[sample_idx]
    print(f"Sample {sample_idx} timestamp: {sample.timestamp}\n")

    # Load market data
    print(f"Loading market data from: {data_dir}")
    from v15.data import load_market_data
    tsla_df, spy_df, vix_df = load_market_data(data_dir)
    print(f"  TSLA: {len(tsla_df)} bars")
    print(f"  SPY: {len(spy_df)} bars\n")

    # Get the appropriate DataFrame
    asset_df = spy_df if asset == 'spy' else tsla_df

    # Build feature keys based on asset
    if asset == 'tsla':
        slope_key = f"{tf}_w{window}_channel_slope"
        intercept_key = f"{tf}_w{window}_channel_intercept"
        std_dev_ratio_key = f"{tf}_w{window}_std_dev_ratio"
    else:  # spy
        slope_key = f"{tf}_w{window}_spy_channel_slope"
        intercept_key = f"{tf}_w{window}_spy_channel_intercept"
        std_dev_ratio_key = f"{tf}_w{window}_spy_std_dev_ratio"

    # Extract features
    print(f"STEP 1: Extract Channel Features")
    print(f"-" * 80)

    if slope_key not in sample.tf_features:
        print(f"ERROR: Feature '{slope_key}' not found in sample")
        print(f"\nAvailable {asset.upper()} channel features:")
        for key in sorted(sample.tf_features.keys()):
            if f"{tf}_w{window}_" in key and asset in key.lower():
                print(f"  {key}: {sample.tf_features[key]:.6f}")
        return

    slope = sample.tf_features[slope_key]
    intercept = sample.tf_features[intercept_key]
    std_dev_ratio = sample.tf_features[std_dev_ratio_key]

    print(f"Feature: {slope_key}")
    print(f"  Value: {slope:.6f}")
    print(f"\nFeature: {intercept_key}")
    print(f"  Value: {intercept:.6f}")
    print(f"\nFeature: {std_dev_ratio_key}")
    print(f"  Value: {std_dev_ratio:.6f}")

    # Check for zero features (indicates failed channel detection)
    if slope == 0.0 and intercept == 0.0 and std_dev_ratio == 0.0:
        print(f"\nWARNING: All features are zero - channel detection likely failed")
        return

    # Get timestamp and find position in DataFrame
    timestamp = sample.timestamp
    if timestamp not in asset_df.index:
        idx_loc = asset_df.index.get_indexer([timestamp], method='nearest')[0]
    else:
        idx_loc = asset_df.index.get_loc(timestamp)

    # Resample to target timeframe if needed
    if tf != '5min':
        from v15.dual_inspector import resample_ohlc
        df_resampled = resample_ohlc(asset_df, tf)

        if timestamp not in df_resampled.index:
            resample_idx = df_resampled.index.get_indexer([timestamp], method='nearest')[0]
        else:
            resample_idx = df_resampled.index.get_loc(timestamp)
    else:
        df_resampled = asset_df
        resample_idx = idx_loc

    # Get the lookback slice (window bars before the timestamp)
    start_idx = max(0, resample_idx - window + 1)
    end_idx = resample_idx + 1
    df_slice = df_resampled.iloc[start_idx:end_idx]

    print(f"\n\nSTEP 2: Get Market Data Slice")
    print(f"-" * 80)
    print(f"Timestamp: {timestamp}")
    print(f"Resample index: {resample_idx}")
    print(f"Slice range: [{start_idx}:{end_idx}] ({len(df_slice)} bars)")

    if len(df_slice) < window:
        print(f"WARNING: Slice has only {len(df_slice)} bars (expected {window})")

    # Calculate average price from the actual data
    avg_price = df_slice['close'].mean()
    print(f"Average price: {avg_price:.2f}")

    # Compute actual std_dev
    std_dev = std_dev_ratio * avg_price
    print(f"Computed std_dev: {std_dev:.6f} (ratio {std_dev_ratio:.6f} * price {avg_price:.2f})")

    # Reconstruct channel bounds
    print(f"\n\nSTEP 3: Reconstruct Channel Bounds")
    print(f"-" * 80)

    x, upper_line, lower_line, center_line = reconstruct_channel_bounds(
        slope, intercept, std_dev_ratio, avg_price, window
    )

    print(f"Reconstructed channel lines (window={window} bars):")
    print(f"\nFirst 5 bars:")
    for i in range(min(5, len(x))):
        print(f"  Bar {i}: center={center_line[i]:.2f}, upper={upper_line[i]:.2f}, lower={lower_line[i]:.2f}")

    print(f"\nLast 5 bars:")
    for i in range(max(0, len(x) - 5), len(x)):
        print(f"  Bar {i}: center={center_line[i]:.2f}, upper={upper_line[i]:.2f}, lower={lower_line[i]:.2f}")

    # Compare with actual market data
    print(f"\n\nSTEP 4: Compare with Actual Market Data")
    print(f"-" * 80)

    print(f"\nPrice range in window:")
    print(f"  Highest high: {df_slice['high'].max():.2f}")
    print(f"  Lowest low: {df_slice['low'].min():.2f}")
    print(f"  Price range: {df_slice['high'].max() - df_slice['low'].min():.2f}")

    print(f"\nChannel bounds range:")
    print(f"  Highest upper: {upper_line.max():.2f}")
    print(f"  Lowest lower: {lower_line.min():.2f}")
    print(f"  Channel range: {upper_line.max() - lower_line.min():.2f}")

    # Check if prices are within channel bounds
    prices_outside = 0
    for i, (idx, row) in enumerate(df_slice.iterrows()):
        if i >= len(upper_line):
            break
        if row['high'] > upper_line[i] or row['low'] < lower_line[i]:
            prices_outside += 1

    print(f"\nPrice containment:")
    print(f"  Bars outside channel: {prices_outside}/{len(df_slice)}")
    print(f"  Containment rate: {(len(df_slice) - prices_outside) / len(df_slice) * 100:.1f}%")

    # Detailed comparison at key positions
    print(f"\n\nSTEP 5: Detailed Position Comparison")
    print(f"-" * 80)

    print(f"\n{'Bar':<6} {'Close':<10} {'Center':<10} {'Upper':<10} {'Lower':<10} {'In Bounds':<10}")
    print(f"{'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for i, (idx, row) in enumerate(df_slice.iterrows()):
        if i >= len(upper_line):
            break

        close_price = row['close']
        in_bounds = lower_line[i] <= close_price <= upper_line[i]
        in_bounds_str = "YES" if in_bounds else "NO"

        # Show first 3, middle 2, and last 3 bars
        if i < 3 or i >= len(df_slice) - 3 or abs(i - len(df_slice)//2) < 1:
            print(f"{i:<6} {close_price:<10.2f} {center_line[i]:<10.2f} {upper_line[i]:<10.2f} {lower_line[i]:<10.2f} {in_bounds_str:<10}")
        elif i == 3:
            print(f"{'...':<6} {'...':<10} {'...':<10} {'...':<10} {'...':<10} {'...':<10}")

    # Summary
    print(f"\n\nSTEP 6: Summary")
    print(f"-" * 80)
    print(f"Features extracted:")
    print(f"  slope = {slope:.6f}")
    print(f"  intercept = {intercept:.6f}")
    print(f"  std_dev_ratio = {std_dev_ratio:.6f}")
    print(f"\nDerived values:")
    print(f"  avg_price = {avg_price:.2f}")
    print(f"  std_dev = {std_dev:.6f}")
    print(f"\nChannel equation:")
    print(f"  center_line[x] = {slope:.6f} * x + {intercept:.6f}")
    print(f"  upper_line[x] = center_line[x] + 2.0 * {std_dev:.6f}")
    print(f"  lower_line[x] = center_line[x] - 2.0 * {std_dev:.6f}")
    print(f"\nVisual check:")
    print(f"  - Blue lines on chart should match these calculated bounds")
    print(f"  - Upper bound at x=0: {upper_line[0]:.2f}")
    print(f"  - Lower bound at x=0: {lower_line[0]:.2f}")
    print(f"  - Upper bound at x={window-1}: {upper_line[-1]:.2f}")
    print(f"  - Lower bound at x={window-1}: {lower_line[-1]:.2f}")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Verify channel bounds match features for a specific sample'
    )
    parser.add_argument(
        '--cache', '-c',
        type=str,
        required=True,
        help='Path to samples pickle file'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        required=True,
        help='Path to data directory with CSV files'
    )
    parser.add_argument(
        '--sample', '-s',
        type=int,
        default=109,
        help='Sample index to verify (default: 109)'
    )
    parser.add_argument(
        '--window', '-w',
        type=int,
        default=10,
        help='Window size to verify (default: 10)'
    )
    parser.add_argument(
        '--asset', '-a',
        type=str,
        default='spy',
        choices=['tsla', 'spy'],
        help='Asset to verify (default: spy)'
    )
    parser.add_argument(
        '--tf', '-t',
        type=str,
        default='5min',
        help='Timeframe to verify (default: 5min)'
    )

    args = parser.parse_args()

    verify_sample(
        args.cache,
        args.data_dir,
        sample_idx=args.sample,
        window=args.window,
        asset=args.asset,
        tf=args.tf
    )
