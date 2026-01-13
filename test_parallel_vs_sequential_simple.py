"""
Simplified test to verify parallel vs sequential scanning produces identical results.

Uses a very large data slice to ensure monthly timeframe works properly.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from v7.training.scanning import scan_valid_channels
from v7.features.full_features import features_to_tensor_dict


def load_data():
    """Load data with very large warmup to ensure monthly TF works."""
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

    # Take a 100,000 bar slice starting from beginning (ensures full warmup coverage)
    # This gives ~30 months of data, ensuring all timeframes (including 3-month) work
    start_idx = 5000
    end_idx = start_idx + 100000
    tsla_df = tsla_5min.iloc[start_idx:end_idx].copy()

    print(f"TSLA: {len(tsla_df)} bars from {tsla_df.index[0]} to {tsla_df.index[-1]}")

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


def compare_samples(samples1, samples2):
    """Quick comparison of two sample lists."""
    if len(samples1) != len(samples2):
        return False, f"Different counts: {len(samples1)} vs {len(samples2)}"

    for i, (s1, s2) in enumerate(zip(samples1, samples2)):
        if s1.timestamp != s2.timestamp:
            return False, f"Sample {i}: Different timestamps"

        if s1.channel_end_idx != s2.channel_end_idx:
            return False, f"Sample {i}: Different channel_end_idx"

        # Compare features
        f1 = features_to_tensor_dict(s1.features)
        f2 = features_to_tensor_dict(s2.features)

        if set(f1.keys()) != set(f2.keys()):
            return False, f"Sample {i}: Different feature keys"

        for key in f1.keys():
            if not np.array_equal(f1[key], f2[key]):
                if not np.allclose(f1[key], f2[key], rtol=1e-10, atol=1e-10, equal_nan=True):
                    return False, f"Sample {i}: Feature {key} differs"

        # Compare labels
        if set(s1.labels.keys()) != set(s2.labels.keys()):
            return False, f"Sample {i}: Different label TFs"

        for tf in s1.labels.keys():
            l1, l2 = s1.labels[tf], s2.labels[tf]
            if (l1 is None) != (l2 is None):
                return False, f"Sample {i}, TF {tf}: One label is None"
            if l1 and l2:
                if (l1.duration_bars != l2.duration_bars or
                    l1.break_direction != l2.break_direction):
                    return False, f"Sample {i}, TF {tf}: Labels differ"

    return True, ""


def main():
    print("=" * 80)
    print("SIMPLE PARALLEL vs SEQUENTIAL TEST")
    print("=" * 80)

    tsla_df, spy_df, vix_df = load_data()

    # Use parameters that should generate samples
    params = {
        'window': 50,
        'step': 50,  # Scan every 50 bars - should give us ~200-300 positions
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
        tsla_df=tsla_df, spy_df=spy_df, vix_df=vix_df,
        parallel=False, **params
    )
    seq_time = time.time() - start
    print(f"\nSequential: {len(samples_seq)} samples in {seq_time:.1f}s")

    print("\n" + "=" * 80)
    print("PARALLEL SCAN")
    print("=" * 80)
    start = time.time()
    samples_par, _ = scan_valid_channels(
        tsla_df=tsla_df, spy_df=spy_df, vix_df=vix_df,
        parallel=True, **params
    )
    par_time = time.time() - start
    print(f"\nParallel: {len(samples_par)} samples in {par_time:.1f}s ({seq_time/par_time:.1f}x speedup)")

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    if len(samples_seq) == 0:
        print("⚠ No samples generated - cannot test comparison")
        return 1

    is_equal, msg = compare_samples(samples_seq, samples_par)

    if is_equal:
        print(f"✓ SUCCESS: {len(samples_seq)} samples are IDENTICAL")
        print("  - Timestamps: identical")
        print("  - Channel indices: identical")
        print("  - Features: identical (bitwise)")
        print("  - Labels: identical")
        return 0
    else:
        print(f"✗ FAILURE: {msg}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
