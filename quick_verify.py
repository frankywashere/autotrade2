#!/usr/bin/env python3
import pickle
import numpy as np
import sys

# Load sample
print("Loading samples...")
with open('v15/cache/production_samples.pkl', 'rb') as f:
    samples = pickle.load(f)

sample = samples[109]
print(f'Sample 109 timestamp: {sample.timestamp}')
print(f'Best window: {sample.best_window}')

# Check for SPY 5min w10 features
window = 10
tf = '5min'
asset = 'spy'

slope_key = f'{tf}_w{window}_spy_channel_slope'
intercept_key = f'{tf}_w{window}_spy_channel_intercept'
std_dev_ratio_key = f'{tf}_w{window}_spy_std_dev_ratio'

print(f'\nFeatures for {asset.upper()} {tf} w{window}:')
if slope_key in sample.tf_features:
    slope = sample.tf_features[slope_key]
    print(f'  {slope_key}: {slope:.6f}')
else:
    print(f'  {slope_key}: NOT FOUND')
    sys.exit(1)

if intercept_key in sample.tf_features:
    intercept = sample.tf_features[intercept_key]
    print(f'  {intercept_key}: {intercept:.6f}')
else:
    print(f'  {intercept_key}: NOT FOUND')
    sys.exit(1)

if std_dev_ratio_key in sample.tf_features:
    std_dev_ratio = sample.tf_features[std_dev_ratio_key]
    print(f'  {std_dev_ratio_key}: {std_dev_ratio:.6f}')
else:
    print(f'  {std_dev_ratio_key}: NOT FOUND')
    sys.exit(1)

# Reconstruct channel bounds
print(f'\n{"="*80}')
print(f'RECONSTRUCTING CHANNEL BOUNDS')
print(f'{"="*80}')

# For SPY 5min w10, we need to get price data to calculate std_dev
# Load SPY data
print('\nLoading SPY data...')
from v15.data import load_market_data
tsla_df, spy_df, vix_df = load_market_data('data')
print(f'SPY data loaded: {len(spy_df)} bars')

# Find the sample timestamp in SPY data
timestamp = sample.timestamp
if timestamp in spy_df.index:
    idx = spy_df.index.get_loc(timestamp)
else:
    idx = spy_df.index.get_indexer([timestamp], method='nearest')[0]

print(f'\nSample timestamp: {timestamp}')
print(f'SPY index: {idx}')

# Get the window slice
start_idx = max(0, idx - window + 1)
end_idx = idx + 1
spy_slice = spy_df.iloc[start_idx:end_idx]

print(f'Window slice: [{start_idx}:{end_idx}] ({len(spy_slice)} bars)')

# Calculate average price
avg_price = spy_slice['close'].mean()
print(f'Average price: {avg_price:.2f}')

# Compute actual std_dev
std_dev = std_dev_ratio * avg_price
print(f'Computed std_dev: {std_dev:.6f} (ratio {std_dev_ratio:.6f} * avg_price {avg_price:.2f})')

# Reconstruct channel lines
x = np.arange(window)
center_line = slope * x + intercept
upper_line = center_line + 2.0 * std_dev
lower_line = center_line - 2.0 * std_dev

print(f'\n{"="*80}')
print(f'CHANNEL BOUNDS')
print(f'{"="*80}')
print(f'\nFirst 3 bars:')
for i in range(min(3, len(x))):
    print(f'  Bar {i}: center={center_line[i]:.2f}, upper={upper_line[i]:.2f}, lower={lower_line[i]:.2f}')

print(f'\nLast 3 bars:')
for i in range(max(0, len(x) - 3), len(x)):
    print(f'  Bar {i}: center={center_line[i]:.2f}, upper={upper_line[i]:.2f}, lower={lower_line[i]:.2f}')

print(f'\n{"="*80}')
print(f'MARKET DATA COMPARISON')
print(f'{"="*80}')

print(f'\nPrice range in window:')
print(f'  Highest high: {spy_slice["high"].max():.2f}')
print(f'  Lowest low: {spy_slice["low"].min():.2f}')

print(f'\nChannel bounds range:')
print(f'  Highest upper: {upper_line.max():.2f}')
print(f'  Lowest lower: {lower_line.min():.2f}')

print(f'\n{"="*80}')
print(f'VISUAL VERIFICATION')
print(f'{"="*80}')
print(f'\nThe blue lines on the chart should be at:')
print(f'  x=0:  upper={upper_line[0]:.2f}, lower={lower_line[0]:.2f}')
print(f'  x={window-1}: upper={upper_line[-1]:.2f}, lower={lower_line[-1]:.2f}')
print(f'\nDone!')
