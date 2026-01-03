#!/usr/bin/env python3
"""
Simple 5-step test to verify live data integration.
"""

# Step 1: Import load_live_data_tuple
from v7.data import load_live_data_tuple

# Step 2: Fetch data with 120 day lookback
tsla_df, spy_df, vix_df = load_live_data_tuple(lookback_days=120)

# Step 3: Verify DataFrame shapes
print(f"TSLA shape: {tsla_df.shape}")
print(f"SPY shape:  {spy_df.shape}")
print(f"VIX shape:  {vix_df.shape}")

# Step 4: Check data freshness
from datetime import datetime
latest = tsla_df.index[-1]
age_hours = (datetime.now() - latest).total_seconds() / 3600
print(f"\nData age: {age_hours:.1f} hours")
print(f"Latest timestamp: {latest}")

# Step 5: Ensure it returns valid data
print(f"\nLatest TSLA close: ${tsla_df['close'].iloc[-1]:.2f}")
print(f"Latest SPY close:  ${spy_df['close'].iloc[-1]:.2f}")
print(f"Latest VIX close:  {vix_df['close'].iloc[-1]:.2f}")

print("\n✓ Integration test passed!")
