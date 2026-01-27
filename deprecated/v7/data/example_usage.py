"""
Example Usage of fetch_live_data()

This demonstrates how dashboard.py should use the fetch_live_data() function.
"""

from fetch_live_data import fetch_live_data

# Fetch data
print("Fetching live data...")
base_df, metadata = fetch_live_data(
    lookback_days=60,
    resample_to='5min',
    enable_live_fetch=True,
    include_multi_resolution=True
)

# Check quality
print(f"\nData Quality: {metadata['quality']}")
print(f"Errors: {len(metadata['errors'])}")
print(f"Warnings: {len(metadata['warnings'])}")

# Access base data
print(f"\nBase TSLA data:")
print(f"  Resolution: 5min")
print(f"  Bars: {len(base_df)}")
print(f"  Range: {base_df.index.min()} to {base_df.index.max()}")
print(f"  Latest price: ${base_df['close'].iloc[-1]:.2f}")

# Access multi-resolution views
multi_res = base_df.attrs['multi_resolution']
print(f"\nMulti-resolution timeframes available:")
for tf in ['5min', '15min', '1h', '4h', 'daily']:
    df = multi_res[tf]
    print(f"  {tf:8s}: {len(df):5d} bars")

# Access SPY data
spy_df = base_df.attrs['spy_df']
print(f"\nSPY data:")
print(f"  Bars: {len(spy_df)}")
print(f"  Latest: ${spy_df['close'].iloc[-1]:.2f}")

# Access VIX data
vix_df = base_df.attrs['vix_df']
print(f"\nVIX data:")
print(f"  Bars: {len(vix_df)}")
print(f"  Latest level: {vix_df['close'].iloc[-1]:.2f}")

# Now you can pass these to your feature extraction
print(f"\n\nReady to extract features:")
print(f"  - extract_full_features(base_df, spy_df, vix_df)")
print(f"  - detect_channel(multi_res['1h'])")
print(f"  - etc.")
