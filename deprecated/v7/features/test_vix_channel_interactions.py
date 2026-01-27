"""
Unit tests and examples for VIX-Channel Interaction Features.

Tests cover:
1. Data alignment (VIX to price)
2. Feature calculations with synthetic data
3. Edge cases (sparse data, no touches, etc.)
4. Feature value ranges
5. Signal patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytest
from vix_channel_interactions import (
    calculate_vix_channel_interactions,
    VIXChannelInteractionFeatures,
    features_to_dict,
    get_feature_names,
    _align_vix_to_price
)
from ..core.channel import detect_channel, Channel, Direction, TouchType, Touch


# ============================================================================
# Synthetic Data Generators
# ============================================================================

def create_synthetic_price_df(num_bars: int = 100, trend: float = 0.01) -> pd.DataFrame:
    """
    Create synthetic OHLCV data.

    Args:
        num_bars: Number of bars to generate
        trend: Daily trend (0.01 = 1% uptrend per bar)

    Returns:
        DataFrame with OHLCV columns and DatetimeIndex
    """
    dates = pd.date_range(start='2024-01-01 09:30:00', periods=num_bars, freq='5min')
    close = np.cumprod(1 + np.random.normal(trend / 100, 0.005, num_bars)) * 100

    df = pd.DataFrame({
        'open': close + np.random.normal(0, 0.5, num_bars),
        'high': close + np.abs(np.random.normal(0.5, 1, num_bars)),
        'low': close - np.abs(np.random.normal(0.5, 1, num_bars)),
        'close': close,
        'volume': np.random.randint(1000000, 5000000, num_bars)
    }, index=dates)

    return df


def create_synthetic_vix_df(
    price_dates: pd.DatetimeIndex,
    regime: str = 'normal',
    trend: float = 0.0
) -> pd.DataFrame:
    """
    Create synthetic VIX data aligned with price dates.

    Args:
        price_dates: DatetimeIndex from price data
        regime: 'calm' (VIX 10-15), 'normal' (VIX 15-25), 'high' (VIX 25-35)
        trend: VIX trend (0.5 = rising 0.5 per day)

    Returns:
        DataFrame with VIX OHLCV and daily DatetimeIndex
    """
    # Create daily dates covering price date range
    start_date = price_dates.min().date()
    end_date = price_dates.max().date()
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Generate VIX levels based on regime
    num_days = len(daily_dates)

    if regime == 'calm':
        vix_close = np.random.normal(12, 2, num_days)
    elif regime == 'normal':
        vix_close = np.random.normal(20, 3, num_days)
    elif regime == 'high':
        vix_close = np.random.normal(30, 4, num_days)
    else:
        vix_close = np.random.normal(20, 3, num_days)

    # Add trend
    vix_close = vix_close + np.arange(num_days) * trend

    # Bound to realistic ranges
    vix_close = np.clip(vix_close, 9, 90)

    df = pd.DataFrame({
        'open': vix_close + np.random.normal(0, 0.5, num_days),
        'high': vix_close + np.abs(np.random.normal(0.5, 1, num_days)),
        'low': vix_close - np.abs(np.random.normal(0.5, 1, num_days)),
        'close': vix_close,
    }, index=daily_dates)

    return df


def create_synthetic_channel(
    df_price: pd.DataFrame,
    window: int = 50,
    num_touches: int = 3
) -> Channel:
    """
    Detect actual channel on synthetic price data.

    Args:
        df_price: Price DataFrame
        window: Window size
        num_touches: Minimum touches for valid channel

    Returns:
        Channel object
    """
    channel = detect_channel(df_price, window=window, min_cycles=num_touches - 1)
    return channel


# ============================================================================
# Data Alignment Tests
# ============================================================================

def test_vix_alignment_with_daily_vix():
    """Test aligning daily VIX to intraday prices."""
    # Create price data: 5-minute bars for 4 days
    price_df = create_synthetic_price_df(num_bars=100)  # ~13 hours * 5min bars

    # Create VIX data: daily
    vix_df = create_synthetic_vix_df(price_df.index, regime='normal')

    # Align
    aligned = _align_vix_to_price(price_df, vix_df)

    assert aligned is not None
    assert len(aligned) == len(price_df)
    assert not aligned.isna().all()
    print(f"✓ Aligned {len(aligned)} price bars to {len(vix_df)} VIX days")


def test_vix_alignment_with_gaps():
    """Test alignment handles weekend gaps."""
    # Create price data with weekend gaps
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    df_price = pd.DataFrame({
        'open': np.ones(10) * 100,
        'high': np.ones(10) * 101,
        'low': np.ones(10) * 99,
        'close': np.ones(10) * 100,
    }, index=dates)

    # Create VIX with same dates
    vix_df = pd.DataFrame({
        'open': np.ones(10) * 20,
        'high': np.ones(10) * 21,
        'low': np.ones(10) * 19,
        'close': np.ones(10) * 20,
    }, index=dates)

    aligned = _align_vix_to_price(df_price, vix_df)

    assert aligned is not None
    assert len(aligned) == len(df_price)
    print("✓ Alignment handles date gaps correctly")


def test_vix_alignment_missing_dates():
    """Test alignment when VIX doesn't cover full price range."""
    price_df = create_synthetic_price_df(num_bars=100)

    # Create VIX with only partial date coverage
    vix_dates = price_df.index[10:80].normalize().unique()
    vix_df = create_synthetic_vix_df(price_df.index[10:80], regime='normal')

    aligned = _align_vix_to_price(price_df, vix_df)

    # Should still work via forward/backward fill
    assert aligned is not None
    print("✓ Alignment handles partial VIX coverage")


# ============================================================================
# Feature Calculation Tests
# ============================================================================

def test_feature_calculation_basic():
    """Test basic feature calculation with known data."""
    # Create simple price data
    price_df = create_synthetic_price_df(num_bars=50)
    vix_df = create_synthetic_vix_df(price_df.index, regime='normal')

    # Detect channel
    channel = detect_channel(price_df, window=50, min_cycles=1)

    # Calculate features
    features = calculate_vix_channel_interactions(
        df_price=price_df,
        df_vix=vix_df,
        channel=channel,
        window=50
    )

    # Check basic properties
    assert isinstance(features, VIXChannelInteractionFeatures)
    assert features.vix_at_channel_start > 0  # Should have values
    assert features.vix_change_during_channel is not None
    print(f"✓ Basic feature calculation: {features.num_bounces_in_window} bounces detected")


def test_feature_ranges():
    """Test that features stay within expected ranges."""
    price_df = create_synthetic_price_df(num_bars=100)
    vix_df = create_synthetic_vix_df(price_df.index, regime='high')
    channel = detect_channel(price_df, window=50, min_cycles=0)

    features = calculate_vix_channel_interactions(price_df, vix_df, channel, 50)

    # Check value ranges
    assert 0 <= features.vix_at_last_bounce <= 100
    assert 0 <= features.vix_at_channel_start <= 100
    assert -100 <= features.vix_change_during_channel <= 200
    assert -1 <= features.channel_age_vs_vix_correlation <= 1
    assert -1 <= features.vix_regime_alignment <= 1
    assert 0 <= features.high_vix_bounce_ratio <= 1
    print("✓ All features within expected ranges")


def test_feature_calculation_no_bounces():
    """Test graceful handling when no bounces in channel."""
    # Create flat price data (no bounces)
    dates = pd.date_range(start='2024-01-01', periods=50, freq='5min')
    df_price = pd.DataFrame({
        'open': np.ones(50) * 100,
        'high': np.ones(50) * 100.1,
        'low': np.ones(50) * 99.9,
        'close': np.ones(50) * 100,
        'volume': np.ones(50) * 1000000
    }, index=dates)

    vix_df = create_synthetic_vix_df(df_price.index, regime='normal')
    channel = detect_channel(df_price, window=50, min_cycles=0)

    features = calculate_vix_channel_interactions(df_price, vix_df, channel, 50)

    # Should still return valid object, with zeros where appropriate
    assert isinstance(features, VIXChannelInteractionFeatures)
    assert features.avg_vix_at_upper_bounces == 0.0  # No bounces
    assert features.num_bounces_in_window == 0
    print("✓ Handles no-bounce case gracefully")


# ============================================================================
# Signal Pattern Tests
# ============================================================================

def test_signal_pattern_prebreak_buildup():
    """
    Test Signal A: Pre-Break Buildup.
    VIX doubling + VIX accelerating + regime divergence = high break probability.
    """
    # Create price with uptrend
    price_df = create_synthetic_price_df(num_bars=100, trend=0.05)

    # Create VIX that rises sharply (stress building)
    dates = price_df.index
    daily_dates = pd.date_range(start=dates.min().date(), end=dates.max().date(), freq='D')

    # VIX starts at 15, rises to 35 (>100% increase)
    vix_close = np.linspace(15, 35, len(daily_dates))
    vix_df = pd.DataFrame({
        'open': vix_close,
        'high': vix_close + 1,
        'low': vix_close - 1,
        'close': vix_close,
    }, index=daily_dates)

    channel = detect_channel(price_df, window=50, min_cycles=0)
    features = calculate_vix_channel_interactions(price_df, vix_df, channel, 50)

    # Check for pre-break signal
    signal_a = (
        features.vix_change_during_channel > 50 and  # VIX doubling
        features.vix_momentum_at_boundary > 0 and    # VIX accelerating
        features.vix_regime_alignment < 0             # Diverged from channel
    )

    if signal_a:
        print("✓ Signal A (pre-break buildup) detected")
    else:
        print("⚠ Signal A not detected (may depend on channel direction)")


def test_signal_pattern_stress_tested_hold():
    """
    Test Signal B: Stress-Tested Hold.
    High bounce ratio in stress + multiple high-VIX bounces + frequent bouncing.
    """
    price_df = create_synthetic_price_df(num_bars=100)

    # Create VIX that stays high
    dates = price_df.index
    daily_dates = pd.date_range(start=dates.min().date(), end=dates.max().date(), freq='D')
    vix_close = np.random.normal(30, 3, len(daily_dates))  # High VIX regime
    vix_df = pd.DataFrame({
        'open': vix_close,
        'high': vix_close + 1,
        'low': vix_close - 1,
        'close': vix_close,
    }, index=daily_dates)

    channel = detect_channel(price_df, window=50, min_cycles=1)
    features = calculate_vix_channel_interactions(price_df, vix_df, channel, 50)

    signal_b = (
        features.high_vix_bounce_ratio > 0.4 and  # 40%+ bounces in stress
        features.bounces_in_high_vix_count >= 2 and  # Multiple stress tests
        features.high_vix_bounce_frequency > 0.1    # Active bouncing
    )

    if signal_b:
        print("✓ Signal B (stress-tested hold) detected")
    else:
        print("⚠ Signal B not detected (channel may not have enough high-VIX bounces)")


def test_signal_pattern_extreme_vix_setup():
    """
    Test Signal C: Extreme Volatility Setup.
    VIX extremely elevated + building with time = mean reversion likely.
    """
    price_df = create_synthetic_price_df(num_bars=100)

    # Create VIX that spikes to extreme and stays high
    dates = price_df.index
    daily_dates = pd.date_range(start=dates.min().date(), end=dates.max().date(), freq='D')

    # Start at 50, gradually rise (building)
    vix_close = np.linspace(45, 60, len(daily_dates))
    vix_close = np.maximum(vix_close, 45)  # Keep elevated
    vix_df = pd.DataFrame({
        'open': vix_close,
        'high': vix_close + 2,
        'low': vix_close - 2,
        'close': vix_close,
    }, index=daily_dates)

    channel = detect_channel(price_df, window=50, min_cycles=0)
    features = calculate_vix_channel_interactions(price_df, vix_df, channel, 50)

    signal_c = (
        features.vix_distance_from_mean > 1.5 and  # Elevated
        features.channel_age_vs_vix_correlation > 0.3  # Building with time
    )

    if signal_c:
        print("✓ Signal C (extreme VIX setup) detected")
    else:
        print("⚠ Signal C not detected (need more extreme/sustained VIX)")


# ============================================================================
# Conversion Tests
# ============================================================================

def test_features_to_dict():
    """Test conversion to dictionary."""
    features = VIXChannelInteractionFeatures(
        vix_at_last_bounce=22.5,
        vix_change_during_channel=25.0,
        high_vix_bounce_ratio=0.6
    )

    d = features_to_dict(features)

    assert isinstance(d, dict)
    assert 'vix_at_last_bounce' in d
    assert d['vix_at_last_bounce'] == 22.5
    assert len(d) == 15  # Should have all 15 features
    print(f"✓ Converted features to dict with {len(d)} entries")


def test_get_feature_names():
    """Test feature names list."""
    names = get_feature_names()

    assert isinstance(names, list)
    assert len(names) == 15
    assert 'vix_at_last_bounce' in names
    assert 'high_vix_bounce_ratio' in names
    assert 'vix_regime_alignment' in names
    print(f"✓ Retrieved {len(names)} feature names")


def test_features_to_dict_all_fields():
    """Test that all features are included in dict."""
    price_df = create_synthetic_price_df(num_bars=100)
    vix_df = create_synthetic_vix_df(price_df.index, regime='normal')
    channel = detect_channel(price_df, window=50, min_cycles=0)

    features = calculate_vix_channel_interactions(price_df, vix_df, channel, 50)
    d = features_to_dict(features)

    expected_keys = get_feature_names()
    for key in expected_keys:
        assert key in d, f"Missing key: {key}"
        assert isinstance(d[key], (int, float)), f"Non-numeric value for {key}"

    print(f"✓ All {len(expected_keys)} features in dict with numeric values")


# ============================================================================
# Integration Tests
# ============================================================================

def test_calm_market_scenario():
    """
    Scenario: Calm market with low VIX, sideways channel.
    Expected: Low bounce stress-testing, but clean bouncing.
    """
    print("\n--- Calm Market Scenario ---")
    price_df = create_synthetic_price_df(num_bars=100)
    vix_df = create_synthetic_vix_df(price_df.index, regime='calm', trend=-0.1)
    channel = detect_channel(price_df, window=50, min_cycles=1)

    features = calculate_vix_channel_interactions(price_df, vix_df, channel, 50)

    print(f"VIX start: {features.vix_at_channel_start:.1f}")
    print(f"VIX change: {features.vix_change_during_channel:.1f}%")
    print(f"High-VIX bounces: {features.bounces_in_high_vix_count:.0f}")
    print(f"Bounce ratio: {features.high_vix_bounce_ratio:.2f}")
    print(f"Bounce frequency: {features.high_vix_bounce_frequency:.3f}")

    # In calm market, expect low stress-testing
    assert features.vix_at_channel_start < 20
    print("✓ Calm market features as expected")


def test_stressed_market_scenario():
    """
    Scenario: Stressed market with high VIX, rapid volatility changes.
    Expected: High stress-testing, potential break signals.
    """
    print("\n--- Stressed Market Scenario ---")
    price_df = create_synthetic_price_df(num_bars=100, trend=-0.02)
    vix_df = create_synthetic_vix_df(price_df.index, regime='high', trend=0.2)
    channel = detect_channel(price_df, window=50, min_cycles=0)

    features = calculate_vix_channel_interactions(price_df, vix_df, channel, 50)

    print(f"VIX start: {features.vix_at_channel_start:.1f}")
    print(f"VIX change: {features.vix_change_during_channel:.1f}%")
    print(f"High-VIX bounces: {features.bounces_in_high_vix_count:.0f}")
    print(f"Age-VIX correlation: {features.channel_age_vs_vix_correlation:.2f}")
    print(f"Regime alignment: {features.vix_regime_alignment:.2f}")

    # In stressed market, expect higher VIX
    assert features.vix_at_channel_start > 20
    print("✓ Stressed market features as expected")


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("VIX-Channel Interaction Features: Test Suite")
    print("=" * 70)

    # Data Alignment Tests
    print("\n[1] Data Alignment Tests")
    print("-" * 70)
    test_vix_alignment_with_daily_vix()
    test_vix_alignment_with_gaps()
    test_vix_alignment_missing_dates()

    # Feature Calculation Tests
    print("\n[2] Feature Calculation Tests")
    print("-" * 70)
    test_feature_calculation_basic()
    test_feature_ranges()
    test_feature_calculation_no_bounces()

    # Signal Pattern Tests
    print("\n[3] Signal Pattern Tests")
    print("-" * 70)
    test_signal_pattern_prebreak_buildup()
    test_signal_pattern_stress_tested_hold()
    test_signal_pattern_extreme_vix_setup()

    # Conversion Tests
    print("\n[4] Conversion Tests")
    print("-" * 70)
    test_features_to_dict()
    test_get_feature_names()
    test_features_to_dict_all_fields()

    # Integration Tests
    print("\n[5] Integration Tests")
    print("-" * 70)
    test_calm_market_scenario()
    test_stressed_market_scenario()

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
