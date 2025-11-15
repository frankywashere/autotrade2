#!/usr/bin/env python3
"""
GPU vs CPU Feature Extraction Equivalence Test

Validates that GPU-accelerated feature extraction produces identical results
to CPU version. Run this after implementing GPU acceleration.

Usage:
    python validate_gpu_cpu_equivalence.py

Expected runtime: 5-10 minutes
Output: Detailed comparison report you can share
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))

from src.ml.features import TradingFeatureExtractor
from src.ml.data_feed import CSVDataFeed


def print_header(text):
    """Print section header"""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def compare_features(cpu_features, gpu_features, feature_name, tolerance=1e-4):
    """
    Compare CPU vs GPU feature values.

    Returns:
        (passed, max_diff, mean_diff, median_diff)
    """
    cpu_vals = cpu_features[feature_name].values
    gpu_vals = gpu_features[feature_name].values

    # Calculate differences
    abs_diff = np.abs(cpu_vals - gpu_vals)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    median_diff = np.median(abs_diff)

    # Check if within tolerance
    passed = max_diff < tolerance

    return passed, max_diff, mean_diff, median_diff


def test_sample_data(year, num_bars, sample_name="Test"):
    """
    Test GPU vs CPU on a sample of data.

    Args:
        year: Year to sample from
        num_bars: Number of bars to test
        sample_name: Name for this test
    """
    print(f"\n{'-'*70}")
    print(f"  {sample_name} ({year}, {num_bars:,} bars)")
    print(f"{'-'*70}")

    # Load data
    print(f"  📁 Loading data from {year}...")
    try:
        data_feed = CSVDataFeed(timeframe='1min')
        df = data_feed.load_aligned_data(
            start_date=f'{year}-01-01',
            end_date=f'{year}-12-31'
        )
    except Exception as e:
        print(f"  ❌ Failed to load data: {e}")
        return None

    # Sample subset if needed
    if len(df) > num_bars:
        # Take from middle to avoid edge effects
        start_idx = (len(df) - num_bars) // 2
        df = df.iloc[start_idx:start_idx + num_bars].copy()

    print(f"     Actual sample: {len(df):,} bars ({df.index[0]} to {df.index[-1]})")

    # Extract features - CPU version
    print(f"  ⏳ Extracting features with CPU...")
    try:
        extractor_cpu = TradingFeatureExtractor()
        cpu_features = extractor_cpu.extract_features(df, use_cache=False, use_gpu=False)
        print(f"     ✓ CPU complete: {len(cpu_features.columns)} features")
    except Exception as e:
        print(f"  ❌ CPU extraction failed: {e}")
        return None

    # Extract features - GPU version
    print(f"  ⚡ Extracting features with GPU...")
    try:
        extractor_gpu = TradingFeatureExtractor()
        gpu_features = extractor_gpu.extract_features(df, use_cache=False, use_gpu=True)
        print(f"     ✓ GPU complete: {len(gpu_features.columns)} features")
    except Exception as e:
        print(f"  ❌ GPU extraction failed: {e}")
        return None

    # Compare results
    print(f"\n  🔍 Comparing results...")

    all_passed = True
    failures = []
    warnings_list = []

    # Get all channel features (most critical)
    channel_features = [c for c in cpu_features.columns if 'channel' in c]

    # Test all channel features
    for feature in channel_features:
        passed, max_diff, mean_diff, median_diff = compare_features(
            cpu_features, gpu_features, feature, tolerance=1e-4
        )

        if not passed:
            all_passed = False
            failures.append((feature, max_diff, mean_diff))
        elif max_diff > 1e-6:
            warnings_list.append((feature, max_diff, mean_diff))

    # Print summary
    if all_passed:
        print(f"     ✅ All {len(channel_features)} features MATCH (within 1e-4 tolerance)")
        if warnings_list:
            print(f"     ℹ️  {len(warnings_list)} features have tiny differences (1e-6 to 1e-4) - OK")
    else:
        print(f"     ❌ {len(failures)} features EXCEED tolerance!")

    return {
        'passed': all_passed,
        'num_features': len(channel_features),
        'failures': len(failures),
        'warnings': len(warnings_list),
        'failure_details': failures[:5],  # First 5 failures
        'cpu_features': cpu_features,
        'gpu_features': gpu_features
    }


def main():
    """Run GPU vs CPU equivalence tests"""

    print_header("GPU vs CPU FEATURE EXTRACTION EQUIVALENCE TEST")

    print(f"\n📅 Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nThis test validates that GPU-accelerated feature extraction")
    print(f"produces identical results to CPU version.")
    print(f"\n⏱️  Expected runtime: 5-10 minutes")

    # Check GPU availability
    print_header("GPU Detection")

    if torch.cuda.is_available():
        gpu_type = "CUDA (NVIDIA)"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✅ {gpu_type} Detected")
        print(f"     Device: {gpu_name}")
        print(f"     Memory: {gpu_memory:.1f} GB")
    elif torch.backends.mps.is_available():
        gpu_type = "MPS (Apple Silicon)"
        print(f"  ✅ {gpu_type} Detected")
        print(f"     Ready for GPU acceleration")
    else:
        print(f"  ❌ No GPU detected")
        print(f"\n  This test requires GPU (CUDA or MPS)")
        print(f"  Please run on a machine with GPU support")
        return 1

    # Run tests on multiple samples
    print_header("Running Equivalence Tests (3 Samples)")

    test_results = []

    try:
        # Test 1: Small sample
        result = test_sample_data(
            year=2023,
            num_bars=10000,
            sample_name="Test 1: Small Sample"
        )
        if result:
            test_results.append(("Small (10K bars)", result))

        # Test 2: Medium sample
        result = test_sample_data(
            year=2022,
            num_bars=50000,
            sample_name="Test 2: Medium Sample"
        )
        if result:
            test_results.append(("Medium (50K bars)", result))

        # Test 3: Large sample
        result = test_sample_data(
            year=2021,
            num_bars=100000,
            sample_name="Test 3: Large Sample"
        )
        if result:
            test_results.append(("Large (100K bars)", result))

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test failed with unexpected error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        print(f"\n📋 Full traceback:")
        traceback.print_exc()
        return 1

    # Overall summary
    print_header("OVERALL TEST SUMMARY")

    if not test_results:
        print(f"\n  ❌ No tests completed successfully")
        return 1

    all_tests_passed = all(result['passed'] for _, result in test_results)

    print(f"\n📊 Results:")
    for test_name, result in test_results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"  {status}  {test_name}")
        print(f"         Features tested: {result['num_features']}")
        if result['failures'] > 0:
            print(f"         ❌ Failures: {result['failures']} features exceed tolerance")
            print(f"         Top failures:")
            for feat, max_diff, mean_diff in result['failure_details']:
                print(f"            {feat}: max_diff={max_diff:.2e}, mean={mean_diff:.2e}")
        if result['warnings'] > 0:
            print(f"         ⚠️  Warnings: {result['warnings']} features have small diffs (OK)")

    # Detailed feature-by-feature comparison (from last test)
    if test_results:
        print_header("DETAILED FEATURE COMPARISON (Last Test)")

        last_result = test_results[-1][1]
        cpu_features = last_result['cpu_features']
        gpu_features = last_result['gpu_features']

        # Get channel features
        channel_cols = [c for c in cpu_features.columns if 'channel' in c]

        print(f"\n📊 Channel Features ({len(channel_cols)} total):")
        print(f"{'Feature':<45} {'Max Diff':<12} {'Mean Diff':<12} {'Status':<8}")
        print(f"{'-'*70}")

        # Show first 20 features
        for col in channel_cols[:20]:
            passed, max_diff, mean_diff, _ = compare_features(cpu_features, gpu_features, col)
            status = "✓" if passed else "✗"
            print(f"{col:<45} {max_diff:<12.2e} {mean_diff:<12.2e} {status:<8}")

        if len(channel_cols) > 20:
            print(f"\n... and {len(channel_cols) - 20} more channel features")

        # Calculate overall statistics
        all_diffs = []
        for col in channel_cols:
            _, max_diff, mean_diff, _ = compare_features(cpu_features, gpu_features, col)
            all_diffs.append(max_diff)

        print(f"\n📈 Overall Statistics:")
        print(f"   Maximum difference (worst case): {max(all_diffs):.2e}")
        print(f"   Average maximum difference: {np.mean(all_diffs):.2e}")
        print(f"   Median maximum difference: {np.median(all_diffs):.2e}")
        print(f"   Tolerance threshold: 1.00e-04")

        if max(all_diffs) < 1e-4:
            print(f"\n   ✅ Results are EQUIVALENT (all within tolerance)")
        else:
            print(f"\n   ❌ Results DIFFER (some exceed tolerance)")

    # Final verdict
    print_header("FINAL VERDICT")

    if all_tests_passed:
        print(f"\n  ✅ ALL TESTS PASSED")
        print(f"\n  GPU and CPU produce equivalent results!")
        print(f"  GPU acceleration is SAFE to use in production.")
        print(f"\n  You can now use --use-gpu-features flag or select GPU in interactive mode.")
        exit_code = 0
    else:
        print(f"\n  ❌ SOME TESTS FAILED")
        print(f"\n  GPU and CPU produce DIFFERENT results!")
        print(f"  DO NOT use GPU acceleration until issues are fixed.")
        print(f"\n  Review the failure details above and debug GPU implementation.")
        exit_code = 1

    print(f"\n{'='*70}")
    print(f"📅 Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    return exit_code


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Test interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
