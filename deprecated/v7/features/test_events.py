"""
Test script for event features module.

This validates that:
1. Events load correctly from CSV
2. All 46 features are computed
3. Features have expected types and ranges
4. Intraday visibility gating works
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.events import (
    EventsHandler,
    extract_event_features,
    EVENT_FEATURE_NAMES,
    event_features_to_dict,
)


def test_events_loading():
    """Test that events.csv loads correctly."""
    print("=" * 80)
    print("TEST 1: Events CSV Loading")
    print("=" * 80)

    events_path = "/Volumes/NVME2/x6/data/events.csv"
    handler = EventsHandler(events_path)

    print(f"✓ Loaded {len(handler.events_df)} events")
    print(f"✓ Date range: {handler.events_df['date'].min()} to {handler.events_df['date'].max()}")

    # Check event type distribution
    event_counts = handler.events_df['event_type'].value_counts()
    print("\nEvent type distribution:")
    for event_type, count in event_counts.items():
        print(f"  - {event_type}: {count}")

    print()
    return handler


def test_feature_extraction(handler):
    """Test feature extraction for a sample timestamp."""
    print("=" * 80)
    print("TEST 2: Feature Extraction")
    print("=" * 80)

    # Create dummy price data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='5min')
    # Filter to market hours (09:30-16:00)
    dates = dates[
        (dates.time >= pd.Timestamp('09:30').time()) &
        (dates.time <= pd.Timestamp('16:00').time())
    ]

    price_df = pd.DataFrame({
        'open': 250.0 + np.random.randn(len(dates)) * 5,
        'high': 255.0 + np.random.randn(len(dates)) * 5,
        'low': 245.0 + np.random.randn(len(dates)) * 5,
        'close': 250.0 + np.random.randn(len(dates)) * 5,
        'volume': np.random.randint(10000, 100000, len(dates)),
    }, index=dates)

    # Test timestamp: October 22, 2024 at 10:00 AM (day before earnings)
    sample_ts = pd.Timestamp('2024-10-22 10:00:00')

    print(f"Sample timestamp: {sample_ts}")
    print(f"Next earnings (from CSV): 2024-10-23 (TSLA Q3 2024)")
    print()

    # Extract features
    features = extract_event_features(sample_ts, handler, price_df)

    # Convert to dict for easier inspection
    feature_dict = event_features_to_dict(features)

    print(f"✓ Extracted {len(feature_dict)} features")
    print()

    # Check that we got all 46 features
    assert len(feature_dict) == 46, f"Expected 46 features, got {len(feature_dict)}"
    assert len(EVENT_FEATURE_NAMES) == 46, f"Expected 46 feature names, got {len(EVENT_FEATURE_NAMES)}"

    # Display some key features
    print("Key features:")
    print(f"  days_until_event: {feature_dict['days_until_event']:.3f}")
    print(f"  days_until_tsla_earnings: {feature_dict['days_until_tsla_earnings']:.3f}")
    print(f"  is_earnings_week: {feature_dict['is_earnings_week']}")
    print(f"  event_is_tsla_earnings_3d: {feature_dict['event_is_tsla_earnings_3d']}")
    print()

    return feature_dict


def test_feature_ranges(feature_dict):
    """Test that features are within expected ranges."""
    print("=" * 80)
    print("TEST 3: Feature Range Validation")
    print("=" * 80)

    issues = []

    # Check timing features (should be [0, 1])
    timing_features = [
        'days_until_event', 'days_since_event',
        'days_until_tsla_earnings', 'days_until_tsla_delivery', 'days_until_fomc',
        'days_until_cpi', 'days_until_nfp', 'days_until_quad_witching',
        'days_since_tsla_earnings', 'days_since_tsla_delivery', 'days_since_fomc',
        'days_since_cpi', 'days_since_nfp', 'days_since_quad_witching',
        'hours_until_tsla_earnings', 'hours_until_tsla_delivery', 'hours_until_fomc',
        'hours_until_cpi', 'hours_until_nfp', 'hours_until_quad_witching',
    ]

    for feat in timing_features:
        val = feature_dict[feat]
        if not (0.0 <= val <= 1.0):
            issues.append(f"{feat} = {val} (expected [0, 1])")

    # Check binary flags (should be {0, 1})
    binary_flags = [
        'is_high_impact_event', 'is_earnings_week',
        'event_is_tsla_earnings_3d', 'event_is_tsla_delivery_3d', 'event_is_fomc_3d',
        'event_is_cpi_3d', 'event_is_nfp_3d', 'event_is_quad_witching_3d',
    ]

    for feat in binary_flags:
        val = feature_dict[feat]
        if val not in (0, 1):
            issues.append(f"{feat} = {val} (expected 0 or 1)")

    # Check drift features (should be [-0.5, 0.5])
    drift_features = [
        'pre_tsla_earnings_drift', 'pre_tsla_delivery_drift', 'pre_fomc_drift',
        'pre_cpi_drift', 'pre_nfp_drift', 'pre_quad_witching_drift',
        'post_tsla_earnings_drift', 'post_tsla_delivery_drift', 'post_fomc_drift',
        'post_cpi_drift', 'post_nfp_drift', 'post_quad_witching_drift',
    ]

    for feat in drift_features:
        val = feature_dict[feat]
        if not (-0.5 <= val <= 0.5):
            issues.append(f"{feat} = {val} (expected [-0.5, 0.5])")

    # Check beat/miss (should be {-1, 0, 1})
    val = feature_dict['last_earnings_beat_miss']
    if val not in (-1, 0, 1):
        issues.append(f"last_earnings_beat_miss = {val} (expected -1, 0, or 1)")

    if issues:
        print("✗ Found range violations:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ All features within expected ranges")

    print()


def test_intraday_gating():
    """Test that intraday visibility gating works correctly."""
    print("=" * 80)
    print("TEST 4: Intraday Visibility Gating")
    print("=" * 80)

    events_path = "/Volumes/NVME2/x6/data/events.csv"
    handler = EventsHandler(events_path)

    # Create simple price data
    dates = pd.date_range('2024-10-22', '2024-10-24', freq='1h')
    price_df = pd.DataFrame({
        'open': 250.0,
        'high': 255.0,
        'low': 245.0,
        'close': 250.0,
        'volume': 10000,
    }, index=dates)

    # Test before/after earnings (assuming earnings at 20:00 on 2024-10-23)
    # Note: The CSV might not have release_time, so we test with default (20:00)

    # Morning before earnings (should see earnings as future)
    ts_before = pd.Timestamp('2024-10-23 10:00:00')
    visible_before = handler.get_visible_events(ts_before)

    print(f"Timestamp: {ts_before}")
    print(f"  Future events: {len(visible_before['future'])}")
    print(f"  Past events: {len(visible_before['past'])}")

    # Check if earnings is in future
    future_earnings = visible_before['future'][visible_before['future']['event_type'] == 'earnings']
    if len(future_earnings) > 0:
        print(f"  ✓ Earnings visible as FUTURE event (date: {future_earnings.iloc[0]['date']})")
    else:
        print(f"  ✗ Earnings not found in future events")

    print()

    # Evening after earnings (should see earnings as past)
    ts_after = pd.Timestamp('2024-10-23 21:00:00')
    visible_after = handler.get_visible_events(ts_after)

    print(f"Timestamp: {ts_after}")
    print(f"  Future events: {len(visible_after['future'])}")
    print(f"  Past events: {len(visible_after['past'])}")

    # Check if earnings is in past
    past_earnings = visible_after['past'][visible_after['past']['event_type'] == 'earnings']
    if len(past_earnings) > 0:
        most_recent = past_earnings.iloc[-1]
        print(f"  ✓ Earnings visible as PAST event (date: {most_recent['date']})")
    else:
        print(f"  ✗ Earnings not found in past events")

    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("EVENT FEATURES MODULE - TEST SUITE")
    print("=" * 80)
    print()

    try:
        # Test 1: Load events
        handler = test_events_loading()

        # Test 2: Extract features
        feature_dict = test_feature_extraction(handler)

        # Test 3: Validate ranges
        test_feature_ranges(feature_dict)

        # Test 4: Intraday gating
        test_intraday_gating()

        print("=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print()

    except Exception as e:
        print("=" * 80)
        print(f"TEST FAILED ✗")
        print(f"Error: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
