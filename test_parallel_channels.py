#!/usr/bin/env python3
"""
Test script to compare parallel vs sequential channel calculation performance.
This verifies that joblib parallelization works and provides speedup.
"""

import time
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.features import TradingFeatureExtractor
from src.ml.data_feed import CSVDataFeed
import config

def test_channel_calculation(use_parallel=True):
    """Test channel calculation with or without parallelization."""

    # Temporarily control parallelization via config
    original_setting = config.PARALLEL_CHANNEL_CALC
    config.PARALLEL_CHANNEL_CALC = use_parallel

    try:
        print(f"\n{'='*60}")
        print(f"Testing {'PARALLEL' if use_parallel else 'SEQUENTIAL'} channel calculation")
        print(f"{'='*60}")

        # Initialize components
        extractor = TradingFeatureExtractor()
        data_feed = CSVDataFeed(data_dir="data", timeframe="1min")

        # Load a small sample of data for testing
        print("Loading data...")
        tsla_df = data_feed.load_data("TSLA", start_date="2017-01-01", end_date="2017-01-31")
        spy_df = data_feed.load_data("SPY", start_date="2017-01-01", end_date="2017-01-31")

        # Merge and prepare data
        tsla_df = tsla_df.add_prefix('tsla_')
        spy_df = spy_df.add_prefix('spy_')
        df = tsla_df.join(spy_df, how='inner')

        print(f"Loaded {len(df)} bars of data")

        # Force cache invalidation for fair comparison
        cache_suffix = f"test_{'parallel' if use_parallel else 'sequential'}_{int(time.time())}"

        # Time the channel extraction
        print(f"\nExtracting channel features ({'parallel' if use_parallel else 'sequential'})...")
        start_time = time.time()

        channel_features = extractor._extract_channel_features(
            df,
            use_cache=False,  # Don't use cache for testing
            use_gpu=False,    # Force CPU mode for parallel test
            cache_suffix=cache_suffix
        )

        elapsed_time = time.time() - start_time

        # Verify results
        print(f"\nResults:")
        print(f"  Features extracted: {len(channel_features.columns)} columns")
        print(f"  Data shape: {channel_features.shape}")
        print(f"  Time elapsed: {elapsed_time:.2f} seconds")
        print(f"  Non-null values: {channel_features.notna().sum().sum()}")

        # Sample some values to ensure they're reasonable
        sample_features = ['tsla_channel_1h_position', 'spy_channel_daily_slope_pct']
        for feat in sample_features:
            if feat in channel_features.columns:
                values = channel_features[feat].dropna()
                if len(values) > 0:
                    print(f"  {feat}: mean={values.mean():.4f}, std={values.std():.4f}")

        return elapsed_time, channel_features

    finally:
        # Restore original setting
        config.PARALLEL_CHANNEL_CALC = original_setting

def main():
    """Main test function."""
    print("="*80)
    print("CHANNEL CALCULATION PARALLELIZATION TEST")
    print("="*80)

    print("\nThis test will:")
    print("1. Run channel calculation in SEQUENTIAL mode")
    print("2. Run channel calculation in PARALLEL mode")
    print("3. Compare results and performance")
    print("\nNote: First run may be slower due to data loading")

    # Test sequential
    seq_time, seq_results = test_channel_calculation(use_parallel=False)

    # Test parallel
    par_time, par_results = test_channel_calculation(use_parallel=True)

    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    # Performance comparison
    speedup = seq_time / par_time if par_time > 0 else 0
    print(f"\nPerformance:")
    print(f"  Sequential time: {seq_time:.2f} seconds")
    print(f"  Parallel time: {par_time:.2f} seconds")
    print(f"  Speedup: {speedup:.2f}x")

    if speedup > 1.5:
        print(f"  ✅ Parallelization provides significant speedup!")
    elif speedup > 1.0:
        print(f"  ⚠️ Modest speedup - dataset may be too small")
    else:
        print(f"  ❌ No speedup - check if parallelization is working")

    # Accuracy comparison
    print(f"\nAccuracy check:")

    # Check if results are identical
    differences = []
    for col in seq_results.columns:
        if col in par_results.columns:
            seq_vals = seq_results[col].values
            par_vals = par_results[col].values

            # Use allclose for floating point comparison
            if not np.allclose(seq_vals, par_vals, rtol=1e-5, atol=1e-8, equal_nan=True):
                max_diff = np.nanmax(np.abs(seq_vals - par_vals))
                differences.append((col, max_diff))

    if not differences:
        print(f"  ✅ Results are identical (within floating point tolerance)")
    else:
        print(f"  ⚠️ Found {len(differences)} columns with differences:")
        for col, diff in differences[:5]:  # Show first 5
            print(f"    - {col}: max diff = {diff:.2e}")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    if speedup > 1.5 and not differences:
        print("✅ Parallelization is working correctly!")
        print(f"   Achieved {speedup:.1f}x speedup with identical results")
    elif speedup > 1.0 and not differences:
        print("⚠️ Parallelization works but with modest speedup")
        print("   Try with larger dataset for better speedup")
    else:
        print("❌ Issues detected - check implementation")

if __name__ == "__main__":
    main()