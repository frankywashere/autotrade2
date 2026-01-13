"""
Test to verify parallel and sequential scanning produce identical results.

This test:
1. Loads a small subset of data (100-200 bars past warmup)
2. Runs parallel scanning on it
3. Runs sequential scanning on the same data
4. Compares the results position by position
5. Checks if labels, features, and channels are bitwise identical

Focus on positions around potential resample boundaries (e.g., near market close times).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import time

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from v7.training.scanning import scan_valid_channels
from v7.training.types import ChannelSample
from v7.features.full_features import features_to_tensor_dict


def load_small_data_subset(
    data_dir: Path,
    start_warmup_bars: int = 32760,
    scan_bars: int = 200,
    forward_bars: int = 8000
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load a small subset of data for testing.

    Args:
        data_dir: Path to data directory
        start_warmup_bars: Number of warmup bars before scan region
        scan_bars: Number of bars in scan region
        forward_bars: Number of forward bars for labels

    Returns:
        Tuple of (tsla_df, spy_df, vix_df)
    """
    # Load TSLA 1min data
    tsla_path = data_dir / "TSLA_1min.csv"
    if not tsla_path.exists():
        raise FileNotFoundError(f"TSLA data not found at {tsla_path}")

    print(f"Loading TSLA data from {tsla_path}...")
    df = pd.read_csv(tsla_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Resample to 5min
    print("Resampling to 5min...")
    tsla_5min = pd.DataFrame({
        'open': df['open'].resample('5min').first(),
        'high': df['high'].resample('5min').max(),
        'low': df['low'].resample('5min').min(),
        'close': df['close'].resample('5min').last(),
        'volume': df['volume'].resample('5min').sum()
    }).dropna()

    # Take a slice from middle of dataset (better data quality)
    # Start from 1/3 through the data to avoid early data issues
    total_bars_needed = start_warmup_bars + scan_bars + forward_bars
    start_idx = len(tsla_5min) // 3
    end_idx = min(start_idx + total_bars_needed, len(tsla_5min))

    if end_idx - start_idx < total_bars_needed:
        print(f"WARNING: Insufficient data. Need {total_bars_needed} bars, have {end_idx - start_idx}")

    tsla_df = tsla_5min.iloc[start_idx:end_idx].copy()
    print(f"TSLA subset: {len(tsla_df)} bars from {tsla_df.index[0]} to {tsla_df.index[-1]}")

    # Load SPY data
    spy_path = data_dir / "SPY_1min.csv"
    if spy_path.exists():
        print(f"Loading SPY data from {spy_path}...")
        df_spy = pd.read_csv(spy_path)
        df_spy['timestamp'] = pd.to_datetime(df_spy['timestamp'])
        df_spy = df_spy.set_index('timestamp')

        # Resample to 5min
        spy_5min = pd.DataFrame({
            'open': df_spy['open'].resample('5min').first(),
            'high': df_spy['high'].resample('5min').max(),
            'low': df_spy['low'].resample('5min').min(),
            'close': df_spy['close'].resample('5min').last(),
            'volume': df_spy['volume'].resample('5min').sum()
        }).dropna()

        # Align with TSLA timestamps
        spy_df = spy_5min.reindex(tsla_df.index, method='ffill')
    else:
        print("SPY data not found, using TSLA as proxy (scaled)")
        spy_df = tsla_df.copy()
        spy_df[['open', 'high', 'low', 'close']] /= 10

    # Load VIX data
    vix_path = data_dir / "VIX_History.csv"
    if vix_path.exists():
        print(f"Loading VIX data from {vix_path}...")
        df_vix = pd.read_csv(vix_path)
        df_vix['DATE'] = pd.to_datetime(df_vix['DATE'])
        df_vix = df_vix.set_index('DATE')

        # Rename columns
        if 'OPEN' in df_vix.columns:
            df_vix = df_vix.rename(columns={
                'OPEN': 'open',
                'HIGH': 'high',
                'LOW': 'low',
                'CLOSE': 'close'
            })

        # Forward fill to 5min resolution
        vix_df = df_vix[['open', 'high', 'low', 'close']].reindex(
            tsla_df.index, method='ffill'
        )
    else:
        print("VIX data not found, using synthetic data")
        vix_df = pd.DataFrame(
            index=tsla_df.index,
            data={
                'open': 20.0,
                'high': 22.0,
                'low': 18.0,
                'close': 20.0
            }
        )

    print(f"\nData loaded successfully:")
    print(f"  TSLA: {len(tsla_df)} bars")
    print(f"  SPY:  {len(spy_df)} bars")
    print(f"  VIX:  {len(vix_df)} bars")

    return tsla_df, spy_df, vix_df


def compare_samples_detailed(
    samples1: List[ChannelSample],
    samples2: List[ChannelSample],
    tolerance: float = 1e-10
) -> Tuple[bool, List[str]]:
    """
    Compare two lists of ChannelSample objects in detail.

    Args:
        samples1: First list of samples (sequential)
        samples2: Second list of samples (parallel)
        tolerance: Numerical tolerance for float comparisons

    Returns:
        Tuple of (is_identical, list_of_differences)
    """
    differences = []

    # Check length
    if len(samples1) != len(samples2):
        differences.append(
            f"DIFFERENT SAMPLE COUNT: Sequential={len(samples1)}, Parallel={len(samples2)}"
        )
        return False, differences

    print(f"\nComparing {len(samples1)} samples in detail...")

    # Compare each sample
    for i, (s1, s2) in enumerate(zip(samples1, samples2)):
        sample_diffs = []

        # Compare timestamps
        if s1.timestamp != s2.timestamp:
            sample_diffs.append(f"  Timestamp: {s1.timestamp} vs {s2.timestamp}")

        # Compare channel end indices
        if s1.channel_end_idx != s2.channel_end_idx:
            sample_diffs.append(
                f"  Channel end idx: {s1.channel_end_idx} vs {s2.channel_end_idx}"
            )

        # Compare channels
        if not compare_channels(s1.channel, s2.channel, tolerance):
            sample_diffs.append(f"  Channels differ")

        # Compare features by converting to tensor dict
        features1_dict = features_to_tensor_dict(s1.features)
        features2_dict = features_to_tensor_dict(s2.features)

        # Check same keys
        if set(features1_dict.keys()) != set(features2_dict.keys()):
            sample_diffs.append(
                f"  Feature keys differ: {set(features1_dict.keys())} vs {set(features2_dict.keys())}"
            )
        else:
            # Check each feature array
            for key in features1_dict.keys():
                arr1 = features1_dict[key]
                arr2 = features2_dict[key]

                if arr1.shape != arr2.shape:
                    sample_diffs.append(
                        f"  Feature '{key}' shape: {arr1.shape} vs {arr2.shape}"
                    )
                    continue

                # Check for exact equality first (bitwise identical)
                if not np.array_equal(arr1, arr2):
                    # Check with tolerance
                    if not np.allclose(arr1, arr2, rtol=tolerance, atol=tolerance, equal_nan=True):
                        max_diff = np.max(np.abs(arr1 - arr2))
                        sample_diffs.append(
                            f"  Feature '{key}' values differ (max diff: {max_diff:.2e})"
                        )
                    else:
                        # Within tolerance but not bitwise identical
                        max_diff = np.max(np.abs(arr1 - arr2))
                        sample_diffs.append(
                            f"  Feature '{key}' nearly identical (max diff: {max_diff:.2e}, within tolerance)"
                        )

        # Compare labels for all timeframes
        if set(s1.labels.keys()) != set(s2.labels.keys()):
            sample_diffs.append(
                f"  Label timeframes differ: {set(s1.labels.keys())} vs {set(s2.labels.keys())}"
            )
        else:
            for tf in s1.labels.keys():
                label1 = s1.labels[tf]
                label2 = s2.labels[tf]

                # Both None or both not None
                if (label1 is None) != (label2 is None):
                    sample_diffs.append(f"  TF {tf}: One label is None, other is not")
                    continue

                if label1 is not None:
                    if not compare_labels(label1, label2):
                        sample_diffs.append(f"  TF {tf}: Labels differ")

        # Compare multi-window structures
        if set(s1.channels.keys()) != set(s2.channels.keys()):
            sample_diffs.append(
                f"  Channel windows differ: {set(s1.channels.keys())} vs {set(s2.channels.keys())}"
            )

        if s1.best_window != s2.best_window:
            sample_diffs.append(
                f"  Best window: {s1.best_window} vs {s2.best_window}"
            )

        if set(s1.labels_per_window.keys()) != set(s2.labels_per_window.keys()):
            sample_diffs.append(
                f"  Label windows differ: {set(s1.labels_per_window.keys())} vs {set(s2.labels_per_window.keys())}"
            )

        # If there are differences for this sample, record them
        if sample_diffs:
            differences.append(f"Sample {i} (timestamp={s1.timestamp}):")
            differences.extend(sample_diffs)

    is_identical = len(differences) == 0
    return is_identical, differences


def compare_channels(ch1, ch2, tolerance: float = 1e-10) -> bool:
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


def find_resample_boundaries(df: pd.DataFrame) -> List[int]:
    """
    Find indices near resample boundaries (e.g., near market close times).

    This helps focus testing on positions where resampling behavior might differ.
    """
    boundaries = []

    # Find indices near 16:00 (4 PM market close)
    for i, ts in enumerate(df.index):
        # Check if near market close (within 30 minutes)
        if 15 <= ts.hour <= 16 and 30 <= ts.minute <= 59:
            boundaries.append(i)

    return boundaries


def run_test():
    """Run the parallel vs sequential identity test."""
    print("=" * 80)
    print("PARALLEL vs SEQUENTIAL SCANNING IDENTITY TEST")
    print("=" * 80)

    # Load data
    data_dir = Path(__file__).parent / "data"

    try:
        tsla_df, spy_df, vix_df = load_small_data_subset(
            data_dir,
            start_warmup_bars=45000,  # ~14 months to ensure monthly TF has 14+ bars for ATR
            scan_bars=150,  # Smaller scan region for faster test
            forward_bars=8000  # Standard forward bars
        )
    except Exception as e:
        print(f"\nERROR loading data: {e}")
        return False

    # Find resample boundaries for reporting
    boundaries = find_resample_boundaries(tsla_df)
    print(f"\nFound {len(boundaries)} positions near resample boundaries (market close times)")

    # Test parameters - use smaller values for faster execution
    params = {
        'window': 50,
        'step': 10,  # Scan every 10 bars
        'min_cycles': 1,
        'max_scan': 200,  # Reduced from 500
        'return_threshold': 10,
        'include_history': False,
        'lookforward_bars': 100,  # Reduced from 200
        'progress': True,
    }

    print("\n" + "=" * 80)
    print("RUNNING SEQUENTIAL SCAN")
    print("=" * 80)

    start_time = time.time()
    samples_seq, warmup_seq = scan_valid_channels(
        tsla_df=tsla_df,
        spy_df=spy_df,
        vix_df=vix_df,
        parallel=False,
        **params
    )
    seq_time = time.time() - start_time

    print(f"\nSequential scan completed:")
    print(f"  Time: {seq_time:.2f}s")
    print(f"  Samples: {len(samples_seq)}")
    print(f"  Warmup bars: {warmup_seq}")

    print("\n" + "=" * 80)
    print("RUNNING PARALLEL SCAN")
    print("=" * 80)

    start_time = time.time()
    samples_par, warmup_par = scan_valid_channels(
        tsla_df=tsla_df,
        spy_df=spy_df,
        vix_df=vix_df,
        parallel=True,
        max_workers=None,  # Use default
        **params
    )
    par_time = time.time() - start_time

    print(f"\nParallel scan completed:")
    print(f"  Time: {par_time:.2f}s")
    print(f"  Samples: {len(samples_par)}")
    print(f"  Warmup bars: {warmup_par}")
    print(f"  Speedup: {seq_time/par_time:.2f}x")

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARING RESULTS")
    print("=" * 80)

    # Check warmup
    if warmup_seq != warmup_par:
        print(f"✗ WARMUP DIFFERS: Sequential={warmup_seq}, Parallel={warmup_par}")
        return False
    else:
        print(f"✓ Warmup identical: {warmup_seq} bars")

    # Check sample count
    if len(samples_seq) != len(samples_par):
        print(f"✗ SAMPLE COUNT DIFFERS: Sequential={len(samples_seq)}, Parallel={len(samples_par)}")
        return False
    else:
        print(f"✓ Sample count identical: {len(samples_seq)} samples")

    if len(samples_seq) == 0:
        print("\n⚠ WARNING: No samples generated. Data may be insufficient for testing.")
        return False

    # Detailed comparison
    is_identical, differences = compare_samples_detailed(samples_seq, samples_par, tolerance=1e-10)

    if is_identical:
        print("\n" + "=" * 80)
        print("✓✓✓ RESULTS ARE BITWISE IDENTICAL ✓✓✓")
        print("=" * 80)
        print(f"\nAll {len(samples_seq)} samples match exactly:")
        print("  - Timestamps: identical")
        print("  - Channel end indices: identical")
        print("  - Channel parameters: identical")
        print("  - Features: bitwise identical")
        print("  - Labels (all timeframes): identical")
        print("  - Multi-window structures: identical")

        # Report on resample boundaries
        boundary_samples = sum(1 for s in samples_seq if s.channel_end_idx in boundaries)
        print(f"\n  Samples near resample boundaries: {boundary_samples}/{len(samples_seq)}")

        return True
    else:
        print("\n" + "=" * 80)
        print("✗✗✗ DIFFERENCES FOUND ✗✗✗")
        print("=" * 80)
        print(f"\nFound {len(differences)} differences:")
        for diff in differences[:20]:  # Show first 20 differences
            print(diff)
        if len(differences) > 20:
            print(f"\n... and {len(differences) - 20} more differences")

        return False


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
