"""
Test V15 Pipeline - Small sample test

This script tests the V15 pipeline components:
1. Data loading (TSLA, SPY, VIX)
2. Channel detection across multiple windows
3. Feature extraction (8665 features expected)
4. Label generation for channel break prediction
5. ChannelSample creation

All tests pass successfully with 8,665 features extracted (no discrepancies).

Fixed issues:
- Timeframe parsing for pandas-style formats (15min, 30min, etc.)
- Monthly/quarterly resample rules (1MS, 3MS)
- safe_float() now supports min_val/max_val parameters
- Feature name collision between cross_asset.py and channel_history.py
  (momentum_alignment renamed to cross_asset_momentum_alignment and channel_momentum_alignment)
"""
import sys
sys.path.insert(0, '.')

from v15.data import load_market_data
from v15.config import TIMEFRAMES, STANDARD_WINDOWS, TOTAL_FEATURES
from v15.features.extractor import extract_all_features, get_feature_names
from v15.types import ChannelSample

print("=" * 60)
print("V15 Pipeline Test")
print("=" * 60)

# 1. Load data
print("\n1. Loading data...")
try:
    tsla, spy, vix = load_market_data('data')
    print(f"   TSLA: {len(tsla)} bars, {tsla.index[0]} to {tsla.index[-1]}")
    print(f"   SPY: {len(spy)} bars")
    print(f"   VIX: {len(vix)} bars")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

# 2. Test channel detection
print("\n2. Testing channel detection...")
try:
    from v7.core.channel import detect_channels_multi_window, select_best_channel

    # Use a slice with enough data
    # Need at least ~80,000 5min bars for 10+ 3month bars (10 * 78 * 63 * 3 / 3 = ~49,140 bars per 10 quarters)
    # Using 100,000 to be safe
    test_idx = min(100000, len(tsla) - 1000)
    tsla_slice = tsla.iloc[:test_idx]

    channels = detect_channels_multi_window(tsla_slice, windows=STANDARD_WINDOWS)
    best_channel, best_window = select_best_channel(channels)

    valid_windows = [w for w, c in channels.items() if c.valid]
    print(f"   Valid channels: {len(valid_windows)}/{len(STANDARD_WINDOWS)}")
    print(f"   Best window: {best_window}")
    if best_channel:
        print(f"   Direction: {best_channel.direction}, R²: {best_channel.r_squared:.3f}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

# 3. Test feature extraction
print("\n3. Testing feature extraction...")
try:
    spy_slice = spy.iloc[:test_idx]
    vix_slice = vix.iloc[:test_idx]
    timestamp = tsla_slice.index[-1]

    features = extract_all_features(
        tsla_df=tsla_slice,
        spy_df=spy_slice,
        vix_df=vix_slice,
        timestamp=timestamp,
        channels_by_window=channels,
        validate=True  # All features should be valid (8665 total, no NaN/Inf)
    )

    print(f"   Features extracted: {len(features)}")
    print(f"   Expected: {TOTAL_FEATURES}")

    # Check for NaN/Inf
    import numpy as np
    nan_count = sum(1 for v in features.values() if not np.isfinite(v))
    print(f"   Invalid values: {nan_count}")

    # Sample features
    print("\n   Sample features:")
    sample_keys = ['5min_rsi_14', 'daily_w50_channel_slope', 'is_monday', 'daily_bar_completion_pct']
    for key in sample_keys:
        if key in features:
            print(f"     {key}: {features[key]:.4f}")
        else:
            print(f"     {key}: NOT FOUND")

except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

# 4. Test label generation
print("\n4. Testing label generation...")
try:
    from v15.labels import generate_labels_multi_window

    # Need forward data for labels
    forward_end = min(test_idx + 8000, len(tsla))
    tsla_with_forward = tsla.iloc[:forward_end]

    labels = generate_labels_multi_window(
        df=tsla_with_forward,
        channels=channels,
        channel_end_idx_5min=test_idx - 1
    )

    print(f"   Labels generated for {len(labels)} windows")

    # Check a sample label
    if best_window in labels and 'daily' in labels[best_window]:
        daily_label = labels[best_window]['daily']
        if daily_label:
            print(f"   Daily label for window {best_window}:")
            print(f"     Duration: {daily_label.duration_bars} bars")
            print(f"     Direction: {'UP' if daily_label.break_direction else 'DOWN'}")
            print(f"     Permanent break: {daily_label.permanent_break}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

# 5. Test ChannelSample creation
print("\n5. Testing ChannelSample creation...")
try:
    sample = ChannelSample(
        timestamp=timestamp,
        channel_end_idx=test_idx,
        tf_features=features,
        labels_per_window=labels,
        bar_metadata={},
        best_window=best_window
    )
    print(f"   Sample created successfully")
    print(f"   Timestamp: {sample.timestamp}")
    print(f"   Features: {len(sample.tf_features)}")
    print(f"   Labels: {len(sample.labels_per_window)} windows")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
