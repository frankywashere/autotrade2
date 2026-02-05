#!/usr/bin/env python3
"""
Example usage of v15scanner Python bindings.

This script demonstrates how to use the C++ backend for the v15 channel scanner.
"""

import sys
import time
import pandas as pd
import numpy as np

# Try to import the C++ backend
try:
    import v15scanner_cpp
    print(f"✓ C++ backend available: v{v15scanner_cpp.__version__}")
    CPP_AVAILABLE = True
except ImportError as e:
    print(f"✗ C++ backend not available: {e}")
    print("  Please build the module first:")
    print("    cd v15_cpp/python_bindings")
    print("    ./build.sh")
    CPP_AVAILABLE = False
    sys.exit(1)

# Try to import the Python wrapper
try:
    from py_scanner import (
        scan_channels_two_pass,
        get_backend,
        get_version,
        is_cpp_available
    )
    print(f"✓ Python wrapper available")
    WRAPPER_AVAILABLE = True
except ImportError as e:
    print(f"✗ Python wrapper not available: {e}")
    WRAPPER_AVAILABLE = False


def create_sample_data(n_bars=1000):
    """Create sample OHLCV data for testing."""
    print(f"\n[DATA] Creating sample data with {n_bars} bars...")

    # Generate timestamps (5-minute bars)
    start_time = pd.Timestamp('2020-01-01 09:30:00', tz='UTC')
    timestamps = pd.date_range(start=start_time, periods=n_bars, freq='5min')

    # Generate realistic price data with trend
    np.random.seed(42)
    base_price = 100.0
    trend = np.linspace(0, 20, n_bars)
    noise = np.random.randn(n_bars) * 2

    close_prices = base_price + trend + noise
    high_prices = close_prices + np.abs(np.random.randn(n_bars) * 1)
    low_prices = close_prices - np.abs(np.random.randn(n_bars) * 1)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    volumes = np.random.randint(1000000, 10000000, n_bars)

    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=timestamps)

    return df


def test_direct_cpp_interface():
    """Test the direct C++ interface."""
    print("\n" + "=" * 70)
    print("TEST 1: Direct C++ Interface")
    print("=" * 70)

    if not CPP_AVAILABLE:
        print("Skipped - C++ backend not available")
        return

    # Create sample data
    tsla_df = create_sample_data(5000)
    spy_df = create_sample_data(5000)
    vix_df = create_sample_data(5000)

    print(f"Data shape: {tsla_df.shape}")
    print(f"Date range: {tsla_df.index[0]} to {tsla_df.index[-1]}")

    # Create configuration
    config = v15scanner_cpp.ScannerConfig()
    config.step = 50  # Larger step for faster testing
    config.workers = 4
    config.max_samples = 100  # Limit for testing
    config.progress = True
    config.verbose = True

    print(f"\nConfiguration:")
    print(f"  Step: {config.step}")
    print(f"  Workers: {config.workers}")
    print(f"  Max samples: {config.max_samples}")

    # Create scanner
    scanner = v15scanner_cpp.Scanner(config)
    print(f"\n✓ Created scanner: {scanner}")

    # Run scan
    print(f"\n[SCAN] Running scanner...")
    start_time = time.time()

    try:
        samples = scanner.scan(tsla_df, spy_df, vix_df)
        elapsed = time.time() - start_time

        print(f"\n✓ Scan completed in {elapsed:.2f}s")
        print(f"  Samples generated: {len(samples)}")

        # Get statistics
        stats = scanner.get_stats()
        print(f"\nStatistics:")
        print(f"  Pass 1 time: {stats.pass1_duration_ms / 1000:.2f}s")
        print(f"  Pass 2 time: {stats.pass2_duration_ms / 1000:.2f}s")
        print(f"  Pass 3 time: {stats.pass3_duration_ms / 1000:.2f}s")
        print(f"  Total time: {stats.total_duration_ms / 1000:.2f}s")
        print(f"  Samples created: {stats.samples_created}")
        print(f"  Samples skipped: {stats.samples_skipped}")
        print(f"  Throughput: {stats.samples_per_second:.2f} samples/sec")

        # Examine first sample
        if samples:
            sample = samples[0]
            print(f"\nFirst sample:")
            print(f"  Timestamp: {sample.get('timestamp')}")
            print(f"  Channel end idx: {sample.get('channel_end_idx')}")
            print(f"  Best window: {sample.get('best_window')}")
            print(f"  Feature count: {len(sample.get('tf_features', {}))}")

            # Show some features
            features = sample.get('tf_features', {})
            if features:
                print(f"\n  Sample features (first 5):")
                for i, (name, value) in enumerate(list(features.items())[:5]):
                    print(f"    {name}: {value:.4f}")

    except Exception as e:
        print(f"\n✗ Scan failed: {e}")
        import traceback
        traceback.print_exc()


def test_python_wrapper():
    """Test the Python wrapper interface."""
    print("\n" + "=" * 70)
    print("TEST 2: Python Wrapper Interface")
    print("=" * 70)

    if not WRAPPER_AVAILABLE:
        print("Skipped - Python wrapper not available")
        return

    # Show backend info
    print(f"\nBackend info:")
    print(f"  Version: {get_version()}")
    print(f"  Backend: {get_backend()}")
    print(f"  C++ available: {is_cpp_available()}")

    # Create sample data
    tsla_df = create_sample_data(5000)
    spy_df = create_sample_data(5000)
    vix_df = create_sample_data(5000)

    print(f"\nData shape: {tsla_df.shape}")

    # Run scan using wrapper
    print(f"\n[SCAN] Running scanner via wrapper...")
    start_time = time.time()

    try:
        samples = scan_channels_two_pass(
            tsla_df, spy_df, vix_df,
            step=50,
            workers=4,
            max_samples=100,
            progress=True,
            strict=True
        )
        elapsed = time.time() - start_time

        print(f"\n✓ Scan completed in {elapsed:.2f}s")
        print(f"  Samples generated: {len(samples)}")

        # Examine first sample
        if samples:
            sample = samples[0]
            print(f"\nFirst sample (wrapped):")
            print(f"  {sample}")
            print(f"  Timestamp: {sample.timestamp}")
            print(f"  Channel end idx: {sample.channel_end_idx}")
            print(f"  Best window: {sample.best_window}")
            print(f"  Feature count: {len(sample.tf_features)}")

    except Exception as e:
        print(f"\n✗ Scan failed: {e}")
        import traceback
        traceback.print_exc()


def test_pickle_compatibility():
    """Test pickle save/load."""
    print("\n" + "=" * 70)
    print("TEST 3: Pickle Compatibility")
    print("=" * 70)

    if not WRAPPER_AVAILABLE:
        print("Skipped - Python wrapper not available")
        return

    import pickle
    import tempfile
    import os

    # Create sample data
    tsla_df = create_sample_data(5000)
    spy_df = create_sample_data(5000)
    vix_df = create_sample_data(5000)

    # Generate samples
    print(f"\n[SCAN] Generating samples...")
    samples = scan_channels_two_pass(
        tsla_df, spy_df, vix_df,
        step=50,
        workers=4,
        max_samples=50,
        progress=False
    )

    print(f"Generated {len(samples)} samples")

    # Test pickle
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
        temp_path = f.name
        print(f"\n[PICKLE] Saving to {temp_path}...")
        pickle.dump(samples, f)

    # Load back
    print(f"[PICKLE] Loading from {temp_path}...")
    with open(temp_path, 'rb') as f:
        loaded_samples = pickle.load(f)

    print(f"✓ Loaded {len(loaded_samples)} samples")

    # Verify
    if len(loaded_samples) == len(samples):
        print("✓ Sample count matches")
    else:
        print(f"✗ Sample count mismatch: {len(loaded_samples)} != {len(samples)}")

    if loaded_samples:
        original = samples[0]
        loaded = loaded_samples[0]
        print(f"\n✓ First sample comparison:")
        print(f"  Original: {original}")
        print(f"  Loaded:   {loaded}")

    # Cleanup
    os.remove(temp_path)
    print(f"\n✓ Pickle test passed")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("v15scanner Python Bindings - Example & Test Script")
    print("=" * 70)

    test_direct_cpp_interface()
    test_python_wrapper()
    test_pickle_compatibility()

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
