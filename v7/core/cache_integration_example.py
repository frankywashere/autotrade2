"""
Example: Integrating FeatureCache with V7 Feature Extraction

Demonstrates how to use the caching layer with existing feature extraction code.
Shows performance improvements in realistic scenarios.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import time
from typing import Dict

from core.cache import FeatureCache, get_global_cache
from core.timeframe import resample_ohlc, TIMEFRAMES
from core.channel import detect_channel, Channel
from features.rsi import calculate_rsi_series, calculate_rsi
from features.full_features import extract_tsla_channel_features, extract_full_features


# ============================================================================
# Example 1: Basic Integration
# ============================================================================

def extract_features_without_cache(df: pd.DataFrame, window: int = 20) -> Dict:
    """Traditional approach - no caching."""
    features = {}

    for tf in TIMEFRAMES[:5]:  # First 5 timeframes
        # Resample
        if tf == '5min':
            df_tf = df
        else:
            df_tf = resample_ohlc(df, tf)

        if len(df_tf) >= window:
            # Detect channel
            channel = detect_channel(df_tf, window=window)

            # Calculate RSI
            rsi_series = calculate_rsi_series(df_tf['close'].values, period=14)
            rsi = calculate_rsi(df_tf['close'].values, period=14)

            features[tf] = {
                'channel': channel,
                'rsi': rsi,
                'rsi_series': rsi_series
            }

    return features


def extract_features_with_cache(
    df: pd.DataFrame,
    window: int = 20,
    cache: FeatureCache = None
) -> Dict:
    """Cached approach - transparently caches expensive operations."""
    if cache is None:
        cache = get_global_cache()

    features = {}

    for tf in TIMEFRAMES[:5]:  # First 5 timeframes
        # Resample (cached)
        if tf == '5min':
            df_tf = df
        else:
            df_tf = cache.resampling.get_or_resample(df, tf, resample_ohlc)

        if len(df_tf) >= window:
            # Detect channel (cached)
            channel = cache.channel.get_or_detect(df_tf, tf, window, detect_channel)

            # Calculate RSI (cached)
            prices = df_tf['close'].values
            rsi_series = cache.rsi.get_or_calculate(
                prices, 14, calculate_rsi_series, 'series'
            )
            rsi = cache.rsi.get_or_calculate(
                prices, 14, calculate_rsi, 'scalar'
            )

            features[tf] = {
                'channel': channel,
                'rsi': rsi,
                'rsi_series': rsi_series
            }

    return features


# ============================================================================
# Example 2: Dataset Generation (Many Repeated Extractions)
# ============================================================================

def generate_training_dataset_without_cache(df: pd.DataFrame, n_samples: int = 100):
    """
    Generate training dataset by extracting features at multiple timepoints.
    WITHOUT caching - slow for repeated data.
    """
    features_list = []
    window = 50

    for i in range(n_samples):
        # Get data up to this point
        end_idx = len(df) - n_samples + i
        df_slice = df.iloc[:end_idx]

        if len(df_slice) >= window:
            features = extract_features_without_cache(df_slice, window)
            features_list.append(features)

    return features_list


def generate_training_dataset_with_cache(df: pd.DataFrame, n_samples: int = 100):
    """
    Generate training dataset by extracting features at multiple timepoints.
    WITH caching - fast due to overlapping data.
    """
    cache = FeatureCache()
    features_list = []
    window = 50

    for i in range(n_samples):
        # Get data up to this point
        end_idx = len(df) - n_samples + i
        df_slice = df.iloc[:end_idx]

        if len(df_slice) >= window:
            features = extract_features_with_cache(df_slice, window, cache)
            features_list.append(features)

    return features_list, cache


# ============================================================================
# Benchmark Comparison
# ============================================================================

def run_benchmark():
    """Compare cached vs non-cached performance."""
    print("\n" + "="*70)
    print("CACHE INTEGRATION BENCHMARK")
    print("="*70)

    # Create realistic dataset
    print("\nGenerating sample data (10,000 5-min bars)...")
    dates = pd.date_range('2024-01-01', periods=10000, freq='5min')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(10000) * 0.5)

    df = pd.DataFrame({
        'open': prices + np.random.randn(10000) * 0.1,
        'high': prices + np.abs(np.random.randn(10000) * 0.3),
        'low': prices - np.abs(np.random.randn(10000) * 0.3),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 10000)
    }, index=dates)

    # ========================================================================
    # Test 1: Single extraction (minimal benefit)
    # ========================================================================
    print("\n" + "-"*70)
    print("TEST 1: Single Feature Extraction")
    print("-"*70)

    start = time.time()
    features1 = extract_features_without_cache(df)
    time_no_cache = time.time() - start
    print(f"Without cache: {time_no_cache*1000:.2f}ms")

    cache = FeatureCache()
    start = time.time()
    features2 = extract_features_with_cache(df, cache=cache)
    time_with_cache = time.time() - start
    print(f"With cache (cold): {time_with_cache*1000:.2f}ms")

    # Verify identical results
    for tf in features1.keys():
        assert np.allclose(
            features1[tf]['channel'].slope,
            features2[tf]['channel'].slope
        ), "Cache corrupted channel!"
        assert np.allclose(
            features1[tf]['rsi_series'],
            features2[tf]['rsi_series'],
            equal_nan=True
        ), "Cache corrupted RSI!"

    print("✓ Cached features are identical to non-cached")

    # ========================================================================
    # Test 2: Repeated extraction (significant benefit)
    # ========================================================================
    print("\n" + "-"*70)
    print("TEST 2: Repeated Extraction (same data)")
    print("-"*70)

    n_repeats = 10

    start = time.time()
    for _ in range(n_repeats):
        _ = extract_features_without_cache(df)
    time_no_cache = time.time() - start
    print(f"Without cache ({n_repeats}x): {time_no_cache*1000:.2f}ms")

    cache.clear()  # Start fresh
    start = time.time()
    for _ in range(n_repeats):
        _ = extract_features_with_cache(df, cache=cache)
    time_with_cache = time.time() - start
    print(f"With cache ({n_repeats}x): {time_with_cache*1000:.2f}ms")
    print(f"Speedup: {time_no_cache/time_with_cache:.1f}x")

    cache.print_stats()

    # ========================================================================
    # Test 3: Dataset generation (overlapping windows)
    # ========================================================================
    print("\n" + "-"*70)
    print("TEST 3: Training Dataset Generation (overlapping windows)")
    print("-"*70)

    n_samples = 50

    print(f"\nGenerating {n_samples} samples without cache...")
    start = time.time()
    dataset1 = generate_training_dataset_without_cache(df, n_samples)
    time_no_cache = time.time() - start
    print(f"Time: {time_no_cache:.2f}s")

    print(f"\nGenerating {n_samples} samples with cache...")
    start = time.time()
    dataset2, cache = generate_training_dataset_with_cache(df, n_samples)
    time_with_cache = time.time() - start
    print(f"Time: {time_with_cache:.2f}s")
    print(f"Speedup: {time_no_cache/time_with_cache:.1f}x")

    cache.print_stats()

    # ========================================================================
    # Test 4: Cache hit rate analysis
    # ========================================================================
    print("\n" + "-"*70)
    print("TEST 4: Cache Hit Rate Analysis")
    print("-"*70)

    cache = FeatureCache()

    # Extract features at many overlapping timepoints
    n_timepoints = 100
    for i in range(n_timepoints):
        end_idx = len(df) - 100 + i
        df_slice = df.iloc[:end_idx]

        if len(df_slice) >= 50:
            _ = extract_features_with_cache(df_slice, 50, cache)

    print(f"\nExtracted features at {n_timepoints} overlapping timepoints")
    cache.print_stats()

    print("\n" + "="*70)
    print("Benchmark complete!")
    print("="*70)


# ============================================================================
# Example 3: Integration with Full Feature Extraction
# ============================================================================

def demonstrate_full_integration():
    """
    Show how to integrate cache with the full feature extraction pipeline.
    """
    print("\n" + "="*70)
    print("FULL PIPELINE INTEGRATION EXAMPLE")
    print("="*70)

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=5000, freq='5min')
    np.random.seed(42)

    tsla_prices = 200 + np.cumsum(np.random.randn(5000) * 1.0)
    spy_prices = 400 + np.cumsum(np.random.randn(5000) * 0.5)
    vix_prices = 15 + np.cumsum(np.random.randn(5000) * 0.3)

    tsla_df = pd.DataFrame({
        'open': tsla_prices + np.random.randn(5000) * 0.2,
        'high': tsla_prices + np.abs(np.random.randn(5000) * 0.5),
        'low': tsla_prices - np.abs(np.random.randn(5000) * 0.5),
        'close': tsla_prices,
        'volume': np.random.randint(10000, 100000, 5000)
    }, index=dates)

    spy_df = pd.DataFrame({
        'open': spy_prices + np.random.randn(5000) * 0.1,
        'high': spy_prices + np.abs(np.random.randn(5000) * 0.3),
        'low': spy_prices - np.abs(np.random.randn(5000) * 0.3),
        'close': spy_prices,
        'volume': np.random.randint(100000, 1000000, 5000)
    }, index=dates)

    # VIX is daily, so resample
    vix_daily_dates = pd.date_range('2024-01-01', periods=100, freq='D')
    vix_df = pd.DataFrame({
        'open': vix_prices[:100] + np.random.randn(100) * 0.1,
        'high': vix_prices[:100] + np.abs(np.random.randn(100) * 0.2),
        'low': vix_prices[:100] - np.abs(np.random.randn(100) * 0.2),
        'close': vix_prices[:100],
        'volume': np.random.randint(1000, 10000, 100)
    }, index=vix_daily_dates)

    print("\nCreated sample TSLA, SPY, and VIX data")

    # Method 1: Without cache (current approach)
    print("\n--- Extracting features WITHOUT cache ---")
    start = time.time()
    features1 = extract_full_features(
        tsla_df, spy_df, vix_df,
        window=50,
        include_history=False  # Disable slow history scan for demo
    )
    time1 = time.time() - start
    print(f"Time: {time1:.3f}s")

    # Method 2: With cache (would need to modify extract_full_features)
    # For now, just demonstrate the concept
    print("\n--- Note: Integrating cache into extract_full_features ---")
    print("""
To integrate caching into extract_full_features():

1. Pass cache instance as optional parameter:
   def extract_full_features(..., cache=None):

2. Use cache for resampling:
   if cache:
       df_tf = cache.resampling.get_or_resample(df, tf, resample_ohlc)
   else:
       df_tf = resample_ohlc(df, tf)

3. Use cache for channel detection:
   if cache:
       channel = cache.channel.get_or_detect(df_tf, tf, window, detect_channel)
   else:
       channel = detect_channel(df_tf, window)

4. Use cache for RSI:
   if cache:
       rsi = cache.rsi.get_or_calculate(prices, period, calculate_rsi, 'scalar')
   else:
       rsi = calculate_rsi(prices, period)

This preserves backward compatibility while enabling caching!
    """)

    print("\n" + "="*70)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    # Run benchmark
    run_benchmark()

    # Show full integration
    demonstrate_full_integration()

    print("\n✓ All integration examples completed successfully!\n")
