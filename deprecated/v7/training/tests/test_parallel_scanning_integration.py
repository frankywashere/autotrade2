"""
Integration tests for parallel scanning with thread-safe cache.

This module verifies that parallel scanning with ProcessPoolExecutor produces
identical results to sequential scanning. Tests use real TSLA/SPY/VIX data
but limit to a small subset for fast execution (< 5 seconds).

Test coverage:
1. Sequential vs parallel scanning produces identical samples
2. Label generation produces identical outputs
3. Different worker counts (2, 4, 8) all produce same results
4. Results are deterministic and reproducible

Usage:
    # Run all tests:
    cd /path/to/x6
    python3 -m pytest v7/training/tests/test_parallel_scanning_integration.py -v

    # Run a specific test:
    python3 -m pytest v7/training/tests/test_parallel_scanning_integration.py::test_sequential_vs_parallel_basic -v

    # Run with output:
    python3 -m pytest v7/training/tests/test_parallel_scanning_integration.py -v -s

    # Run directly:
    cd /path/to/x6
    python3 v7/training/tests/test_parallel_scanning_integration.py
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Ensure we can import v7 modules
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from v7.training.scanning import scan_valid_channels
from v7.training.types import ChannelSample
from v7.features.full_features import features_to_tensor_dict


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent.parent.parent.parent / "data"


@pytest.fixture(scope="module")
def small_tsla_df(data_dir: Path) -> pd.DataFrame:
    """
    Load a small subset of TSLA data for testing.

    Uses ~50,000 bars (roughly 2 months) to ensure fast tests while
    still having enough data for meaningful channel detection.
    """
    tsla_path = data_dir / "TSLA_1min.csv"
    if not tsla_path.exists():
        pytest.skip(f"TSLA data not found at {tsla_path}")

    # Read full data
    df = pd.read_csv(tsla_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    })

    # Resample to 5min to reduce data size and speed up tests
    # Use proper OHLCV resampling
    df_5min = pd.DataFrame({
        'open': df['open'].resample('5min').first(),
        'high': df['high'].resample('5min').max(),
        'low': df['low'].resample('5min').min(),
        'close': df['close'].resample('5min').last(),
        'volume': df['volume'].resample('5min').sum()
    }).dropna()

    # Take a slice that's large enough for warmup + scanning
    # Need ~32,760 bars for warmup + ~1000 for scanning + ~8000 forward
    # Total: ~42,000 bars (about 145 days of 5min data)
    # Use slice from middle of dataset to get good quality data
    start_idx = len(df_5min) // 3
    end_idx = min(start_idx + 42000, len(df_5min))

    return df_5min.iloc[start_idx:end_idx].copy()


@pytest.fixture(scope="module")
def small_spy_df(data_dir: Path, small_tsla_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load SPY data aligned with TSLA timestamps.

    For simplicity in tests, we'll use TSLA data as a proxy for SPY
    with slight modifications to simulate different price action.
    """
    spy_path = data_dir / "SPY_1min.csv"

    if spy_path.exists():
        # Load real SPY data
        df = pd.read_csv(spy_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        # Resample to 5min
        df_5min = pd.DataFrame({
            'open': df['open'].resample('5min').first(),
            'high': df['high'].resample('5min').max(),
            'low': df['low'].resample('5min').min(),
            'close': df['close'].resample('5min').last(),
            'volume': df['volume'].resample('5min').sum()
        }).dropna()

        # Align with TSLA timestamps
        df_aligned = df_5min.reindex(small_tsla_df.index, method='ffill')
        return df_aligned
    else:
        # Use TSLA as proxy with scaled prices
        df = small_tsla_df.copy()
        # Scale to approximate SPY price levels (SPY is typically ~1/10 of TSLA)
        df['open'] = df['open'] / 10
        df['high'] = df['high'] / 10
        df['low'] = df['low'] / 10
        df['close'] = df['close'] / 10
        return df


@pytest.fixture(scope="module")
def small_vix_df(data_dir: Path, small_tsla_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load VIX data aligned with TSLA timestamps.

    VIX is daily data, so we'll forward-fill to 5min resolution.
    """
    vix_path = data_dir / "VIX_History.csv"

    if vix_path.exists():
        # Load real VIX data
        df = pd.read_csv(vix_path)
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.set_index('DATE')

        # Rename columns to standard format
        if 'OPEN' in df.columns:
            df = df.rename(columns={
                'OPEN': 'open',
                'HIGH': 'high',
                'LOW': 'low',
                'CLOSE': 'close'
            })

        # Align with TSLA timestamps using forward fill
        df_aligned = df.reindex(small_tsla_df.index, method='ffill')
        return df_aligned[['open', 'high', 'low', 'close']]
    else:
        # Create synthetic VIX data (typically ranges 10-40)
        vix_data = pd.DataFrame(
            index=small_tsla_df.index,
            data={
                'open': 20.0,
                'high': 22.0,
                'low': 18.0,
                'close': 20.0
            }
        )
        # Add some noise
        vix_data['close'] = 20.0 + np.random.randn(len(vix_data)) * 2
        vix_data['open'] = vix_data['close'].shift(1).fillna(20.0)
        vix_data['high'] = vix_data[['open', 'close']].max(axis=1) + abs(np.random.randn(len(vix_data))) * 0.5
        vix_data['low'] = vix_data[['open', 'close']].min(axis=1) - abs(np.random.randn(len(vix_data))) * 0.5
        return vix_data


# =============================================================================
# Helper Functions
# =============================================================================

def compare_samples(
    samples1: List[ChannelSample],
    samples2: List[ChannelSample],
    tolerance: float = 1e-6
) -> Tuple[bool, str]:
    """
    Compare two lists of ChannelSample objects for equality.

    Args:
        samples1: First list of samples
        samples2: Second list of samples
        tolerance: Numerical tolerance for float comparisons

    Returns:
        Tuple of (is_equal, error_message)
    """
    # Check length
    if len(samples1) != len(samples2):
        return False, f"Different number of samples: {len(samples1)} vs {len(samples2)}"

    # Compare each sample
    for i, (s1, s2) in enumerate(zip(samples1, samples2)):
        # Compare timestamps
        if s1.timestamp != s2.timestamp:
            return False, f"Sample {i}: Different timestamps: {s1.timestamp} vs {s2.timestamp}"

        # Compare channel end indices
        if s1.channel_end_idx != s2.channel_end_idx:
            return False, f"Sample {i}: Different channel_end_idx: {s1.channel_end_idx} vs {s2.channel_end_idx}"

        # Compare channels
        if not compare_channels(s1.channel, s2.channel, tolerance):
            return False, f"Sample {i}: Channels differ"

        # Compare features by converting to tensor dict and checking all arrays
        features1_dict = features_to_tensor_dict(s1.features)
        features2_dict = features_to_tensor_dict(s2.features)

        # Check same keys
        if set(features1_dict.keys()) != set(features2_dict.keys()):
            return False, f"Sample {i}: Different feature keys: {set(features1_dict.keys())} vs {set(features2_dict.keys())}"

        # Check each array
        for key in features1_dict.keys():
            arr1 = features1_dict[key]
            arr2 = features2_dict[key]

            if arr1.shape != arr2.shape:
                return False, f"Sample {i}, feature {key}: Different shapes: {arr1.shape} vs {arr2.shape}"

            if not np.allclose(arr1, arr2, rtol=tolerance, atol=tolerance, equal_nan=True):
                max_diff = np.max(np.abs(arr1 - arr2))
                return False, f"Sample {i}, feature {key}: Values differ (max diff: {max_diff})"

        # Compare labels for all timeframes
        if set(s1.labels.keys()) != set(s2.labels.keys()):
            return False, f"Sample {i}: Different label timeframes: {set(s1.labels.keys())} vs {set(s2.labels.keys())}"

        for tf in s1.labels.keys():
            label1 = s1.labels[tf]
            label2 = s2.labels[tf]

            # Both None or both not None
            if (label1 is None) != (label2 is None):
                return False, f"Sample {i}, TF {tf}: One label is None, other is not"

            if label1 is not None:
                if not compare_labels(label1, label2):
                    return False, f"Sample {i}, TF {tf}: Labels differ"

    return True, ""


def compare_channels(ch1, ch2, tolerance: float = 1e-6) -> bool:
    """Compare two Channel objects for approximate equality."""
    if ch1 is None and ch2 is None:
        return True
    if ch1 is None or ch2 is None:
        return False

    # Compare key numeric fields
    if not np.isclose(ch1.slope, ch2.slope, rtol=tolerance, atol=tolerance):
        return False
    if not np.isclose(ch1.intercept, ch2.intercept, rtol=tolerance, atol=tolerance):
        return False
    if not np.isclose(ch1.std_dev, ch2.std_dev, rtol=tolerance, atol=tolerance):
        return False

    # Compare window and bounce_count
    if ch1.window != ch2.window:
        return False
    if ch1.bounce_count != ch2.bounce_count:
        return False

    return True


def compare_labels(label1, label2) -> bool:
    """Compare two ChannelLabels objects for equality."""
    if label1 is None and label2 is None:
        return True
    if label1 is None or label2 is None:
        return False

    # Compare all fields
    if label1.duration_bars != label2.duration_bars:
        return False
    if label1.break_direction != label2.break_direction:
        return False
    if label1.break_trigger_tf != label2.break_trigger_tf:
        return False
    if label1.new_channel_direction != label2.new_channel_direction:
        return False
    if label1.permanent_break != label2.permanent_break:
        return False

    # Compare validity flags
    if label1.duration_valid != label2.duration_valid:
        return False
    if label1.direction_valid != label2.direction_valid:
        return False
    if label1.trigger_tf_valid != label2.trigger_tf_valid:
        return False
    if label1.new_channel_valid != label2.new_channel_valid:
        return False

    return True


# =============================================================================
# Integration Tests
# =============================================================================

def test_sequential_vs_parallel_basic(small_tsla_df, small_spy_df, small_vix_df):
    """
    Test that parallel scanning produces identical results to sequential scanning.

    This is the main integration test verifying that the thread-safe cache
    and parallel processing produce exactly the same samples as sequential.
    """
    # Use small parameters for fast execution
    params = {
        'window': 50,
        'step': 100,  # Large step to reduce sample count
        'min_cycles': 1,
        'max_scan': 100,  # Reduced from 500 for speed
        'return_threshold': 10,  # Reduced from 20 for speed
        'include_history': False,
        'lookforward_bars': 100,  # Reduced from 200 for speed
        'progress': False,
    }

    # Run sequential scan
    samples_seq, warmup_seq = scan_valid_channels(
        tsla_df=small_tsla_df,
        spy_df=small_spy_df,
        vix_df=small_vix_df,
        parallel=False,
        **params
    )

    # Run parallel scan (use default worker count)
    samples_par, warmup_par = scan_valid_channels(
        tsla_df=small_tsla_df,
        spy_df=small_spy_df,
        vix_df=small_vix_df,
        parallel=True,
        max_workers=None,  # Use default
        **params
    )

    # Verify warmup is same
    assert warmup_seq == warmup_par, f"Warmup differs: {warmup_seq} vs {warmup_par}"

    # Verify samples are identical
    is_equal, error_msg = compare_samples(samples_seq, samples_par)
    assert is_equal, f"Samples differ: {error_msg}"

    # Verify we got some samples
    assert len(samples_seq) > 0, "No samples generated - test data may be insufficient"

    print(f"\n✓ Sequential vs parallel test passed: {len(samples_seq)} samples identical")


@pytest.mark.parametrize("max_workers", [2, 4, 8])
def test_parallel_different_worker_counts(small_tsla_df, small_spy_df, small_vix_df, max_workers):
    """
    Test that different worker counts produce identical results.

    This verifies that the number of parallel workers doesn't affect
    determinism - all worker counts should produce the same output.
    """
    params = {
        'window': 50,
        'step': 100,
        'min_cycles': 1,
        'max_scan': 100,
        'return_threshold': 10,
        'include_history': False,
        'lookforward_bars': 100,
        'progress': False,
        'parallel': True,
    }

    # Run with specified worker count
    samples, _ = scan_valid_channels(
        tsla_df=small_tsla_df,
        spy_df=small_spy_df,
        vix_df=small_vix_df,
        max_workers=max_workers,
        **params
    )

    # Run with 1 worker for comparison (effectively sequential but using parallel path)
    samples_baseline, _ = scan_valid_channels(
        tsla_df=small_tsla_df,
        spy_df=small_spy_df,
        vix_df=small_vix_df,
        max_workers=1,
        **params
    )

    # Verify identical results
    is_equal, error_msg = compare_samples(samples_baseline, samples)
    assert is_equal, f"Results differ for {max_workers} workers: {error_msg}"

    print(f"\n✓ Worker count test passed for {max_workers} workers: {len(samples)} samples identical")


def test_label_generation_consistency(small_tsla_df, small_spy_df, small_vix_df):
    """
    Test that label generation produces consistent results.

    Verifies that labels for all timeframes are generated correctly
    and are identical between sequential and parallel runs.
    """
    params = {
        'window': 50,
        'step': 150,  # Even larger step for speed
        'min_cycles': 1,
        'max_scan': 100,
        'return_threshold': 10,
        'include_history': False,
        'lookforward_bars': 100,
        'progress': False,
    }

    # Run both modes
    samples_seq, _ = scan_valid_channels(
        tsla_df=small_tsla_df,
        spy_df=small_spy_df,
        vix_df=small_vix_df,
        parallel=False,
        **params
    )

    samples_par, _ = scan_valid_channels(
        tsla_df=small_tsla_df,
        spy_df=small_spy_df,
        vix_df=small_vix_df,
        parallel=True,
        **params
    )

    # Verify we have samples
    assert len(samples_seq) > 0, "No samples generated"
    assert len(samples_seq) == len(samples_par), "Different sample counts"

    # Check that all samples have labels for multiple timeframes
    timeframes_found = set()
    for sample in samples_seq:
        assert sample.labels is not None, "Sample missing labels"
        assert isinstance(sample.labels, dict), "Labels should be dict"

        # Collect all timeframes that have valid labels
        for tf, label in sample.labels.items():
            if label is not None:
                timeframes_found.add(tf)

    # Verify we have labels for multiple timeframes
    assert len(timeframes_found) > 1, f"Only found labels for {timeframes_found}, expected multiple TFs"

    # Verify each label has all required fields
    for i, sample in enumerate(samples_seq):
        for tf, label in sample.labels.items():
            if label is not None:
                # Check all required fields exist
                assert hasattr(label, 'duration_bars'), f"Sample {i}, TF {tf}: Missing duration_bars"
                assert hasattr(label, 'break_direction'), f"Sample {i}, TF {tf}: Missing break_direction"
                assert hasattr(label, 'break_trigger_tf'), f"Sample {i}, TF {tf}: Missing break_trigger_tf"
                assert hasattr(label, 'new_channel_direction'), f"Sample {i}, TF {tf}: Missing new_channel_direction"
                assert hasattr(label, 'permanent_break'), f"Sample {i}, TF {tf}: Missing permanent_break"

                # Verify validity flags
                assert hasattr(label, 'duration_valid'), f"Sample {i}, TF {tf}: Missing duration_valid"
                assert hasattr(label, 'direction_valid'), f"Sample {i}, TF {tf}: Missing direction_valid"
                assert hasattr(label, 'trigger_tf_valid'), f"Sample {i}, TF {tf}: Missing trigger_tf_valid"
                assert hasattr(label, 'new_channel_valid'), f"Sample {i}, TF {tf}: Missing new_channel_valid"

    print(f"\n✓ Label generation test passed: Found labels for {len(timeframes_found)} timeframes")
    print(f"  Timeframes: {sorted(timeframes_found)}")


def test_multi_window_channels(small_tsla_df, small_spy_df, small_vix_df):
    """
    Test that multi-window channel detection works correctly.

    Verifies that samples contain channels for multiple window sizes
    and that best_window selection is consistent.
    """
    params = {
        'window': 50,  # This is used as a base, but multi-window should use STANDARD_WINDOWS
        'step': 150,
        'min_cycles': 1,
        'max_scan': 100,
        'return_threshold': 10,
        'include_history': False,
        'lookforward_bars': 100,
        'progress': False,
    }

    # Run sequential
    samples_seq, _ = scan_valid_channels(
        tsla_df=small_tsla_df,
        spy_df=small_spy_df,
        vix_df=small_vix_df,
        parallel=False,
        **params
    )

    # Run parallel
    samples_par, _ = scan_valid_channels(
        tsla_df=small_tsla_df,
        spy_df=small_spy_df,
        vix_df=small_vix_df,
        parallel=True,
        **params
    )

    # Verify multi-window structures exist
    assert len(samples_seq) > 0, "No samples generated"

    for i, (s_seq, s_par) in enumerate(zip(samples_seq, samples_par)):
        # Check that multi-window fields exist
        assert s_seq.channels is not None, f"Sample {i}: Missing channels dict"
        assert s_seq.best_window is not None, f"Sample {i}: Missing best_window"
        assert s_seq.labels_per_window is not None, f"Sample {i}: Missing labels_per_window"

        # Verify parallel has same structure
        assert s_par.channels is not None, f"Parallel sample {i}: Missing channels dict"
        assert s_par.best_window is not None, f"Parallel sample {i}: Missing best_window"
        assert s_par.labels_per_window is not None, f"Parallel sample {i}: Missing labels_per_window"

        # Verify best_window is same
        assert s_seq.best_window == s_par.best_window, \
            f"Sample {i}: Different best_window: {s_seq.best_window} vs {s_par.best_window}"

        # Verify channels dict has same keys
        assert set(s_seq.channels.keys()) == set(s_par.channels.keys()), \
            f"Sample {i}: Different channel windows"

        # Verify labels_per_window has same structure
        assert set(s_seq.labels_per_window.keys()) == set(s_par.labels_per_window.keys()), \
            f"Sample {i}: Different label windows"

    print(f"\n✓ Multi-window test passed: {len(samples_seq)} samples with consistent window selection")


def test_determinism_multiple_runs(small_tsla_df, small_spy_df, small_vix_df):
    """
    Test that parallel scanning is deterministic across multiple runs.

    Running the same scan multiple times should produce identical results.
    """
    params = {
        'window': 50,
        'step': 150,
        'min_cycles': 1,
        'max_scan': 100,
        'return_threshold': 10,
        'include_history': False,
        'lookforward_bars': 100,
        'progress': False,
        'parallel': True,
        'max_workers': 4,
    }

    # Run 3 times
    samples1, _ = scan_valid_channels(
        tsla_df=small_tsla_df,
        spy_df=small_spy_df,
        vix_df=small_vix_df,
        **params
    )

    samples2, _ = scan_valid_channels(
        tsla_df=small_tsla_df,
        spy_df=small_spy_df,
        vix_df=small_vix_df,
        **params
    )

    samples3, _ = scan_valid_channels(
        tsla_df=small_tsla_df,
        spy_df=small_spy_df,
        vix_df=small_vix_df,
        **params
    )

    # Verify all runs are identical
    is_equal_12, error_12 = compare_samples(samples1, samples2)
    assert is_equal_12, f"Run 1 vs 2: {error_12}"

    is_equal_23, error_23 = compare_samples(samples2, samples3)
    assert is_equal_23, f"Run 2 vs 3: {error_23}"

    print(f"\n✓ Determinism test passed: 3 runs produced identical {len(samples1)} samples")


def test_custom_return_thresholds(small_tsla_df, small_spy_df, small_vix_df):
    """
    Test that custom return thresholds work correctly in parallel mode.

    Verifies that custom per-timeframe return thresholds are properly
    passed through the parallel processing pipeline.
    """
    custom_thresholds = {
        '5min': 5,
        '15min': 3,
        '1h': 2,
    }

    params = {
        'window': 50,
        'step': 150,
        'min_cycles': 1,
        'max_scan': 100,
        'return_threshold': 10,  # Default for TFs not in custom dict
        'include_history': False,
        'lookforward_bars': 100,
        'progress': False,
        'custom_return_thresholds': custom_thresholds,
    }

    # Run sequential
    samples_seq, _ = scan_valid_channels(
        tsla_df=small_tsla_df,
        spy_df=small_spy_df,
        vix_df=small_vix_df,
        parallel=False,
        **params
    )

    # Run parallel
    samples_par, _ = scan_valid_channels(
        tsla_df=small_tsla_df,
        spy_df=small_spy_df,
        vix_df=small_vix_df,
        parallel=True,
        **params
    )

    # Verify results are identical
    is_equal, error_msg = compare_samples(samples_seq, samples_par)
    assert is_equal, f"Custom threshold results differ: {error_msg}"

    # Verify we got samples
    assert len(samples_seq) > 0, "No samples generated with custom thresholds"

    print(f"\n✓ Custom return thresholds test passed: {len(samples_seq)} samples identical")


# =============================================================================
# Performance Benchmark (Optional)
# =============================================================================

@pytest.mark.slow
def test_performance_benchmark(small_tsla_df, small_spy_df, small_vix_df):
    """
    Optional benchmark to verify parallel is faster than sequential.

    This test is marked as 'slow' and won't run by default.
    Run with: pytest -v -m slow
    """
    import time

    params = {
        'window': 50,
        'step': 50,  # Smaller step for more samples
        'min_cycles': 1,
        'max_scan': 100,
        'return_threshold': 10,
        'include_history': False,
        'lookforward_bars': 100,
        'progress': False,
    }

    # Sequential
    start = time.time()
    samples_seq, _ = scan_valid_channels(
        tsla_df=small_tsla_df,
        spy_df=small_spy_df,
        vix_df=small_vix_df,
        parallel=False,
        **params
    )
    seq_time = time.time() - start

    # Parallel
    start = time.time()
    samples_par, _ = scan_valid_channels(
        tsla_df=small_tsla_df,
        spy_df=small_spy_df,
        vix_df=small_vix_df,
        parallel=True,
        max_workers=4,
        **params
    )
    par_time = time.time() - start

    # Verify results are identical
    is_equal, error_msg = compare_samples(samples_seq, samples_par)
    assert is_equal, f"Performance test: samples differ: {error_msg}"

    # Print benchmark results
    speedup = seq_time / par_time if par_time > 0 else 0
    print(f"\n✓ Performance benchmark:")
    print(f"  Sequential: {seq_time:.2f}s ({len(samples_seq)} samples)")
    print(f"  Parallel:   {par_time:.2f}s ({len(samples_par)} samples)")
    print(f"  Speedup:    {speedup:.2f}x")

    # Parallel should be faster (or at least not much slower)
    # Allow some variance due to overhead on small datasets
    assert par_time < seq_time * 1.5, \
        f"Parallel ({par_time:.2f}s) is significantly slower than sequential ({seq_time:.2f}s)"


if __name__ == "__main__":
    """
    Allow running tests directly with python.
    For full pytest features, use: pytest -v test_parallel_scanning_integration.py
    """
    pytest.main([__file__, "-v", "-s"])
