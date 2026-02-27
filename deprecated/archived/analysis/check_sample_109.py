#!/usr/bin/env python3
"""
Quick check of sample 109, window 10, SPY 5min channel bounds.
This script simulates what the inspector does to reconstruct the channel.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 5min OHLCV data to target timeframe."""
    if timeframe == '5min':
        return df

    rule_map = {
        '15min': '15min', '30min': '30min', '1h': '1h',
        '2h': '2h', '3h': '3h', '4h': '4h',
        'daily': '1D', 'weekly': '1W', 'monthly': 'ME'
    }

    rule = rule_map.get(timeframe)
    if not rule:
        return df

    # For monthly resampling, try 'ME' first (pandas 2.2+), fall back to 'M' (legacy)
    if timeframe == 'monthly':
        try:
            return df.resample(rule).agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum'
            }).dropna()
        except ValueError:
            rule = 'M'

    return df.resample(rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()


# Configuration
sample_idx = 109
window = 10
asset = 'spy'
tf = '5min'

print(f"{'='*80}")
print(f"CHECKING SAMPLE {sample_idx}, WINDOW {window}, {asset.upper()} {tf}")
print(f"{'='*80}\n")

# Load samples
cache_path = 'v15/cache/production_samples.pkl'
print(f"Loading samples from: {cache_path}")
with open(cache_path, 'rb') as f:
    samples = pickle.load(f)
print(f"  Loaded {len(samples)} samples\n")

sample = samples[sample_idx]
print(f"Sample {sample_idx}:")
print(f"  Timestamp: {sample.timestamp}")
print(f"  Best window: {sample.best_window}\n")

# Build feature keys
if asset == 'tsla':
    slope_key = f"{tf}_w{window}_channel_slope"
    intercept_key = f"{tf}_w{window}_channel_intercept"
    std_dev_ratio_key = f"{tf}_w{window}_std_dev_ratio"
else:  # spy
    slope_key = f"{tf}_w{window}_spy_channel_slope"
    intercept_key = f"{tf}_w{window}_spy_channel_intercept"
    std_dev_ratio_key = f"{tf}_w{window}_spy_std_dev_ratio"

# Extract features
print(f"Feature extraction:")
if slope_key not in sample.tf_features:
    print(f"  ERROR: {slope_key} not found!")
    # Show available features
    print(f"\n  Available {asset} {tf} w{window} features:")
    for key in sorted(sample.tf_features.keys()):
        if f"{tf}_w{window}" in key and (asset in key or asset == 'tsla'):
            print(f"    {key}: {sample.tf_features[key]:.6f}")
    exit(1)

slope = sample.tf_features[slope_key]
intercept = sample.tf_features[intercept_key]
std_dev_ratio = sample.tf_features[std_dev_ratio_key]

print(f"  {slope_key}: {slope:.6f}")
print(f"  {intercept_key}: {intercept:.6f}")
print(f"  {std_dev_ratio_key}: {std_dev_ratio:.6f}")

# Check for zero features
if slope == 0.0 and intercept == 0.0 and std_dev_ratio == 0.0:
    print(f"\n  WARNING: All features are zero - channel detection likely failed")
    exit(0)

# Load market data
print(f"\nLoading market data from: data/")
from v15.data import load_market_data
tsla_df, spy_df, vix_df = load_market_data('data')

asset_df = spy_df if asset == 'spy' else tsla_df
print(f"  {asset.upper()}: {len(asset_df)} bars\n")

# Get timestamp position
timestamp = sample.timestamp
if timestamp not in asset_df.index:
    idx_loc = asset_df.index.get_indexer([timestamp], method='nearest')[0]
else:
    idx_loc = asset_df.index.get_loc(timestamp)

# Resample if needed
df_resampled = resample_ohlc(asset_df, tf)

# Find position in resampled data
if timestamp not in df_resampled.index:
    resample_idx = df_resampled.index.get_indexer([timestamp], method='nearest')[0]
else:
    resample_idx = df_resampled.index.get_loc(timestamp)

# Get lookback slice
start_idx = max(0, resample_idx - window + 1)
end_idx = resample_idx + 1
df_slice = df_resampled.iloc[start_idx:end_idx]

print(f"Data slice:")
print(f"  Timestamp: {timestamp}")
print(f"  Resample index: {resample_idx}")
print(f"  Slice range: [{start_idx}:{end_idx}] ({len(df_slice)} bars)\n")

# Calculate average price
avg_price = df_slice['close'].mean()
std_dev = std_dev_ratio * avg_price

print(f"Price calculations:")
print(f"  Average close: {avg_price:.2f}")
print(f"  std_dev = {std_dev_ratio:.6f} * {avg_price:.2f} = {std_dev:.6f}\n")

# Reconstruct channel lines
x = np.arange(window)
center_line = slope * x + intercept
upper_line = center_line + 2.0 * std_dev
lower_line = center_line - 2.0 * std_dev

print(f"{'='*80}")
print(f"RECONSTRUCTED CHANNEL BOUNDS")
print(f"{'='*80}\n")

print(f"Channel equation:")
print(f"  center_line[x] = {slope:.6f} * x + {intercept:.6f}")
print(f"  upper_line[x] = center_line[x] + 2.0 * {std_dev:.6f}")
print(f"  lower_line[x] = center_line[x] - 2.0 * {std_dev:.6f}\n")

print(f"First 3 bars:")
for i in range(min(3, len(x))):
    print(f"  Bar {i}: center={center_line[i]:.2f}, upper={upper_line[i]:.2f}, lower={lower_line[i]:.2f}")

print(f"\nLast 3 bars:")
for i in range(max(0, len(x) - 3), len(x)):
    print(f"  Bar {i}: center={center_line[i]:.2f}, upper={upper_line[i]:.2f}, lower={lower_line[i]:.2f}")

print(f"\n{'='*80}")
print(f"COMPARISON WITH MARKET DATA")
print(f"{'='*80}\n")

print(f"Price range:")
print(f"  Highest high: {df_slice['high'].max():.2f}")
print(f"  Lowest low: {df_slice['low'].min():.2f}")

print(f"\nChannel range:")
print(f"  Highest upper: {upper_line.max():.2f}")
print(f"  Lowest lower: {lower_line.min():.2f}")

# Check containment
prices_outside = 0
for i, (idx, row) in enumerate(df_slice.iterrows()):
    if i >= len(upper_line):
        break
    if row['high'] > upper_line[i] or row['low'] < lower_line[i]:
        prices_outside += 1

containment_pct = (len(df_slice) - prices_outside) / len(df_slice) * 100
print(f"\nContainment:")
print(f"  Bars outside channel: {prices_outside}/{len(df_slice)}")
print(f"  Containment rate: {containment_pct:.1f}%")

print(f"\n{'='*80}")
print(f"WHAT TO CHECK ON THE CHART")
print(f"{'='*80}\n")

print(f"The BLUE LINES on the chart should be at:")
print(f"  At x=0 (first bar):")
print(f"    Upper bound: {upper_line[0]:.2f}")
print(f"    Lower bound: {lower_line[0]:.2f}")
print(f"  At x={window-1} (last bar, before projection):")
print(f"    Upper bound: {upper_line[-1]:.2f}")
print(f"    Lower bound: {lower_line[-1]:.2f}")

print(f"\nThe channel should {'appear correct' if containment_pct >= 80 else 'NOT contain most prices (CHECK FOR BUG!)'}.")
print(f"\n{'='*80}\n")
