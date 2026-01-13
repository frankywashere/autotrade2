"""
Comprehensive test comparing parallel vs sequential scanning.

This test:
1. Uses 50,000 bars of data (enough for timeframes up to weekly)
2. Runs both parallel and sequential with identical parameters
3. Compares results position by position
4. Specifically checks positions around market close times
5. Reports any differences

Monthly and 3-month timeframes are automatically skipped due to insufficient history.
Tests after lookahead fix is applied.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time
from datetime import datetime, time as dt_time

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from v7.training.scanning import scan_valid_channels
from v7.features.full_features import features_to_tensor_dict


def load_data():
    """Load 50K bars (enough for weekly TF, monthly will be skipped)."""
    print("Loading TSLA data...")
    df = pd.read_csv("data/TSLA_1min.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Resample to 5min
    tsla_5min = pd.DataFrame({
        'open': df['open'].resample('5min').first(),
        'high': df['high'].resample('5min').max(),
        'low': df['low'].resample('5min').min(),
        'close': df['close'].resample('5min').last(),
        'volume': df['volume'].resample('5min').sum()
    }).dropna()

    # Use 50K bars total - enough for weekly (390 bars), not enough for monthly (1638 bars)
    # This will cause monthly/3month TFs to be automatically skipped
    start_idx = 5000  # Skip early data
    end_idx = start_idx + 50000
    if end_idx > len(tsla_5min):
        end_idx = len(tsla_5min)
        print(f"Warning: Only {len(tsla_5min)} bars available, using {end_idx - start_idx}")

    tsla_df = tsla_5min.iloc[start_idx:end_idx].copy()
    print(f"TSLA: {len(tsla_df)} bars from {tsla_df.index[0]} to {tsla_df.index[-1]}")
    print(f"  This provides ~128 weekly bars (sufficient) but only ~30 monthly bars (insufficient for ATR)")
    print(f"  Monthly and 3-month TFs will be automatically skipped")

    # Load SPY
    print("Loading SPY data...")
    df_spy = pd.read_csv("data/SPY_1min.csv")
    df_spy['timestamp'] = pd.to_datetime(df_spy['timestamp'])
    df_spy = df_spy.set_index('timestamp')

    spy_5min = pd.DataFrame({
        'open': df_spy['open'].resample('5min').first(),
        'high': df_spy['high'].resample('5min').max(),
        'low': df_spy['low'].resample('5min').min(),
        'close': df_spy['close'].resample('5min').last(),
        'volume': df_spy['volume'].resample('5min').sum()
    }).dropna()

    spy_df = spy_5min.reindex(tsla_df.index, method='ffill')

    # Load VIX
    print("Loading VIX data...")
    df_vix = pd.read_csv("data/VIX_History.csv")
    df_vix['DATE'] = pd.to_datetime(df_vix['DATE'])
    df_vix = df_vix.set_index('DATE')

    if 'OPEN' in df_vix.columns:
        df_vix = df_vix.rename(columns={
            'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close'
        })

    vix_df = df_vix[['open', 'high', 'low', 'close']].reindex(
        tsla_df.index, method='ffill'
    )

    print(f"Data loaded: TSLA={len(tsla_df)}, SPY={len(spy_df)}, VIX={len(vix_df)}")
    return tsla_df, spy_df, vix_df


def is_near_market_close(timestamp):
    """Check if timestamp is near market close (15:30-16:00 ET)."""
    # Convert to time only
    t = timestamp.time()
    # Market close is 16:00, check if within 30 minutes before
    return dt_time(15, 30) <= t <= dt_time(16, 0)


def compare_features(f1, f2, sample_idx, key):
    """Compare two feature tensors, return (is_equal, error_msg)."""
    if not np.array_equal(f1, f2):
        if not np.allclose(f1, f2, rtol=1e-10, atol=1e-10, equal_nan=True):
            # Compute statistics about the difference
            diff = np.abs(f1 - f2)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            return False, f"Sample {sample_idx}: Feature '{key}' differs (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})"
    return True, ""


def compare_labels(l1, l2, sample_idx, tf):
    """Compare two labels, return (is_equal, error_msg)."""
    if (l1 is None) != (l2 is None):
        return False, f"Sample {sample_idx}, TF {tf}: One label is None (seq={l1 is not None}, par={l2 is not None})"

    if l1 and l2:
        if l1.duration_bars != l2.duration_bars:
            return False, f"Sample {sample_idx}, TF {tf}: duration_bars differs (seq={l1.duration_bars}, par={l2.duration_bars})"
        if l1.break_direction != l2.break_direction:
            return False, f"Sample {sample_idx}, TF {tf}: break_direction differs (seq={l1.break_direction}, par={l2.break_direction})"

    return True, ""


def compare_samples_detailed(samples1, samples2, tsla_df):
    """
    Detailed comparison of two sample lists.

    Returns: (is_equal, error_messages, market_close_positions)
    """
    if len(samples1) != len(samples2):
        return False, [f"Different counts: {len(samples1)} vs {len(samples2)}"], []

    errors = []
    market_close_positions = []

    for i, (s1, s2) in enumerate(zip(samples1, samples2)):
        # Check if this position is near market close
        is_close_time = is_near_market_close(s1.timestamp)
        if is_close_time:
            market_close_positions.append(i)

        # Compare timestamps
        if s1.timestamp != s2.timestamp:
            errors.append(f"Sample {i}: Different timestamps (seq={s1.timestamp}, par={s2.timestamp})")
            continue

        # Compare channel end index
        if s1.channel_end_idx != s2.channel_end_idx:
            errors.append(f"Sample {i}: Different channel_end_idx (seq={s1.channel_end_idx}, par={s2.channel_end_idx})")

        # Compare features
        f1 = features_to_tensor_dict(s1.features)
        f2 = features_to_tensor_dict(s2.features)

        if set(f1.keys()) != set(f2.keys()):
            errors.append(f"Sample {i}: Different feature keys (seq={set(f1.keys())}, par={set(f2.keys())})")
            continue

        for key in f1.keys():
            is_equal, msg = compare_features(f1[key], f2[key], i, key)
            if not is_equal:
                errors.append(msg)

        # Compare labels
        if set(s1.labels.keys()) != set(s2.labels.keys()):
            errors.append(f"Sample {i}: Different label TFs (seq={set(s1.labels.keys())}, par={set(s2.labels.keys())})")
            continue

        for tf in s1.labels.keys():
            is_equal, msg = compare_labels(s1.labels[tf], s2.labels[tf], i, tf)
            if not is_equal:
                errors.append(msg)

        # Limit error reporting to first 10 errors
        if len(errors) >= 10:
            errors.append("... (truncated after 10 errors)")
            break

    return len(errors) == 0, errors, market_close_positions


def main():
    print("=" * 80)
    print("COMPREHENSIVE PARALLEL vs SEQUENTIAL TEST")
    print("Testing after lookahead fix with 50K bars (weekly TF, monthly skipped)")
    print("=" * 80)

    tsla_df, spy_df, vix_df = load_data()

    # Use all 50K bars - monthly TFs will automatically fail and be skipped
    # which is fine, we're testing up to weekly TF
    tsla_scan = tsla_df
    spy_scan = spy_df
    vix_scan = vix_df

    print(f"Scanning all {len(tsla_scan)} bars")

    # Use parameters that should generate samples
    # Step of 100 to get ~500 test positions
    params = {
        'window': 50,
        'step': 100,
        'min_cycles': 1,
        'max_scan': 200,
        'return_threshold': 10,
        'include_history': False,
        'lookforward_bars': 100,
        'progress': True,
    }

    print("\n" + "=" * 80)
    print("SEQUENTIAL SCAN")
    print("=" * 80)
    start = time.time()
    samples_seq, _ = scan_valid_channels(
        tsla_df=tsla_scan, spy_df=spy_scan, vix_df=vix_scan,
        parallel=False, **params
    )
    seq_time = time.time() - start
    print(f"\nSequential: {len(samples_seq)} samples in {seq_time:.1f}s")

    print("\n" + "=" * 80)
    print("PARALLEL SCAN")
    print("=" * 80)
    start = time.time()
    samples_par, _ = scan_valid_channels(
        tsla_df=tsla_scan, spy_df=spy_scan, vix_df=vix_scan,
        parallel=True, **params
    )
    par_time = time.time() - start
    print(f"\nParallel: {len(samples_par)} samples in {par_time:.1f}s ({seq_time/par_time:.1f}x speedup)")

    print("\n" + "=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)

    if len(samples_seq) == 0:
        print("WARNING: No samples generated - cannot test comparison")
        return 1

    is_equal, errors, market_close_positions = compare_samples_detailed(samples_seq, samples_par, tsla_scan)

    print(f"\nTotal samples compared: {len(samples_seq)}")
    print(f"Samples near market close (15:30-16:00): {len(market_close_positions)}")

    if market_close_positions:
        print(f"Market close sample indices: {market_close_positions[:10]}" +
              (f" ... and {len(market_close_positions)-10} more" if len(market_close_positions) > 10 else ""))

    if is_equal:
        print("\n" + "=" * 80)
        print("SUCCESS: ALL SAMPLES ARE IDENTICAL")
        print("=" * 80)
        print(f"  Verified {len(samples_seq)} samples")
        print("  - Timestamps: IDENTICAL")
        print("  - Channel indices: IDENTICAL")
        print("  - Features: IDENTICAL (numerical tolerance: 1e-10)")
        print("  - Labels: IDENTICAL")
        print(f"  - Market close positions checked: {len(market_close_positions)}")
        print("\nParallel implementation is correct and produces identical results!")
        return 0
    else:
        print("\n" + "=" * 80)
        print("FAILURE: DIFFERENCES FOUND")
        print("=" * 80)
        print(f"\nFound {len(errors)} error(s):\n")
        for error in errors:
            print(f"  {error}")

        # Check if any errors occurred at market close positions
        market_close_errors = [e for e in errors if any(f"Sample {pos}:" in e for pos in market_close_positions)]
        if market_close_errors:
            print("\n" + "=" * 80)
            print("ERRORS AT MARKET CLOSE TIMES:")
            print("=" * 80)
            for error in market_close_errors:
                print(f"  {error}")

        return 1


if __name__ == "__main__":
    sys.exit(main())
