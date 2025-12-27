"""
Pre-compute Targets for Fast Training (v5.9.4)

This script pre-computes:
1. Breakout labels (Fix #1) - eliminates linear regression per sample
2. Sample-indexed target arrays (Fix #3) - eliminates 2,223 dict insertions per sample

Run this ONCE after feature extraction. The pre-computed files are used by
HierarchicalDataset for ~10-17 minute/epoch speedup.

Usage:
    python -m src.ml.precompute_targets --cache-dir data/feature_cache

The script reads existing cached files and creates NEW files alongside them.
No existing cache files are modified.
"""

import numpy as np
import pandas as pd
import pickle
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import sys

# Add parent to path for imports
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

import config
from src.ml.features import HIERARCHICAL_TIMEFRAMES


def compute_channel_breakout(
    past_prices: np.ndarray,
    future_prices: np.ndarray,
    current_price: float,
    lookback: int = 60,
    channel_std: float = 2.0,
    breakout_threshold: float = 1.0
) -> Dict[str, float]:
    """
    Compute channel breakout labels (same logic as HierarchicalDataset._detect_channel_breakout).

    This is the expensive linear regression that was running per-sample.
    Now it runs once during pre-computation.
    """
    # Default values (no breakout)
    result = {
        'breakout_occurred': 0.0,
        'breakout_direction': 0.5,
        'breakout_bars_log': np.log(len(future_prices) + 1) if len(future_prices) > 0 else 0.0,
        'breakout_magnitude': 0.0
    }

    # Need enough past data
    if len(past_prices) < lookback or len(past_prices) < 10:
        return result

    if len(future_prices) == 0:
        return result

    # Calculate channel from past prices (last `lookback` bars)
    y = past_prices[-lookback:]
    X = np.arange(lookback)

    # Fit linear regression
    X_mean = X.mean()
    y_mean = y.mean()
    denominator = np.sum((X - X_mean) ** 2)
    if denominator < 1e-10:
        return result

    slope = np.sum((X - X_mean) * (y - y_mean)) / (denominator + 1e-10)
    intercept = y_mean - slope * X_mean

    # Calculate residuals and channel width
    fitted = slope * X + intercept
    residuals = y - fitted
    channel_width = np.std(residuals)
    if channel_width < 1e-10:
        return result

    # Project channel forward
    upper_bounds = []
    lower_bounds = []
    for i in range(len(future_prices)):
        future_bar_idx = lookback + i
        projected_center = slope * future_bar_idx + intercept
        upper_bounds.append(projected_center + channel_std * channel_width)
        lower_bounds.append(projected_center - channel_std * channel_width)

    upper_bounds = np.array(upper_bounds)
    lower_bounds = np.array(lower_bounds)

    # Detect breakout
    breakout_threshold_dist = breakout_threshold * channel_width

    for i, price in enumerate(future_prices):
        # Check for upward breakout
        if price > upper_bounds[i] + breakout_threshold_dist:
            result['breakout_occurred'] = 1.0
            result['breakout_direction'] = 1.0
            result['breakout_bars_log'] = np.log(i + 1 + 1e-6)
            result['breakout_magnitude'] = (price - upper_bounds[i]) / channel_width
            break
        # Check for downward breakout
        elif price < lower_bounds[i] - breakout_threshold_dist:
            result['breakout_occurred'] = 1.0
            result['breakout_direction'] = 0.0
            result['breakout_bars_log'] = np.log(i + 1 + 1e-6)
            result['breakout_magnitude'] = (lower_bounds[i] - price) / channel_width
            break

    return result


def load_existing_cache(cache_dir: Path) -> Dict:
    """Load existing cached files needed for pre-computation."""
    print("Loading existing cache files...")

    cache = {}

    # Find tf_meta file
    tf_meta_files = sorted(cache_dir.glob("tf_meta_*.json"))
    if not tf_meta_files:
        raise FileNotFoundError(f"No tf_meta_*.json found in {cache_dir}")

    tf_meta_path = tf_meta_files[-1]
    print(f"  Loading {tf_meta_path.name}")
    with open(tf_meta_path) as f:
        cache['tf_meta'] = json.load(f)
    cache['cache_key'] = cache['tf_meta']['cache_key']

    # Load 5min sequence and timestamps (for breakout computation)
    cache_key = cache['cache_key']

    seq_5min_path = cache_dir / f"tf_sequence_5min_{cache_key}.npy"
    ts_5min_path = cache_dir / f"tf_timestamps_5min_{cache_key}.npy"

    if seq_5min_path.exists():
        print(f"  Loading {seq_5min_path.name}")
        cache['tf_5min'] = np.load(str(seq_5min_path), mmap_mode='r')
        cache['tf_5min_shape'] = cache['tf_5min'].shape
    else:
        raise FileNotFoundError(f"Missing {seq_5min_path}")

    if ts_5min_path.exists():
        print(f"  Loading {ts_5min_path.name}")
        cache['ts_5min'] = np.load(str(ts_5min_path))
    else:
        raise FileNotFoundError(f"Missing {ts_5min_path}")

    # Find close price column index in 5min features
    columns_5min = cache['tf_meta']['timeframe_columns']['5min']
    close_idx = None
    for i, col in enumerate(columns_5min):
        if col == 'tsla_close' or 'close' in col.lower():
            close_idx = i
            break
    if close_idx is None:
        raise ValueError("Cannot find close price column in 5min features")
    cache['close_idx'] = close_idx
    print(f"  Close price column index: {close_idx}")

    # Load continuation labels for all timeframes
    cache['cont_labels'] = {}
    cache['cont_ts_to_idx'] = {}

    for tf in HIERARCHICAL_TIMEFRAMES:
        pattern = f"continuation_labels_{tf}_*.pkl"
        matching = sorted(cache_dir.glob(pattern))
        if matching:
            label_path = matching[-1]
            print(f"  Loading {label_path.name}")
            with open(label_path, 'rb') as f:
                df = pickle.load(f)
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                cache['cont_labels'][tf] = df
                # Build timestamp -> row_idx lookup
                ts_to_idx = {}
                for i, ts in enumerate(df.index):
                    ts_ns = pd.Timestamp(ts).value
                    ts_to_idx[ts_ns] = i
                cache['cont_ts_to_idx'][tf] = ts_to_idx

    print(f"  Loaded continuation labels for {len(cache['cont_labels'])} timeframes")

    # Load transition labels for all timeframes
    cache['trans_labels'] = {}
    cache['trans_ts_to_idx'] = {}

    for tf in HIERARCHICAL_TIMEFRAMES:
        pattern = f"transition_labels_{tf}_*.pkl"
        matching = sorted(cache_dir.glob(pattern))
        if matching:
            label_path = matching[-1]
            print(f"  Loading {label_path.name}")
            with open(label_path, 'rb') as f:
                df = pickle.load(f)
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                cache['trans_labels'][tf] = df
                # Build timestamp -> row_idx lookup
                ts_to_idx = {}
                for i, ts in enumerate(df.index):
                    ts_ns = pd.Timestamp(ts).value
                    ts_to_idx[ts_ns] = i
                cache['trans_ts_to_idx'][tf] = ts_to_idx

    print(f"  Loaded transition labels for {len(cache['trans_labels'])} timeframes")

    # Load raw OHLC for future prices (breakout computation)
    raw_ohlc_path = parent_dir / "data" / "tsla_1min_data.csv"
    if raw_ohlc_path.exists():
        print(f"  Loading raw OHLC from {raw_ohlc_path.name}")
        raw_df = pd.read_csv(raw_ohlc_path, parse_dates=['timestamp'], index_col='timestamp')
        cache['raw_ohlc'] = raw_df[['tsla_open', 'tsla_high', 'tsla_low', 'tsla_close']].values
        cache['raw_ohlc_timestamps'] = raw_df.index.values.astype('datetime64[ns]').astype('int64')
    else:
        print(f"  WARNING: Raw OHLC not found at {raw_ohlc_path}, breakout labels will use 5min closes")
        cache['raw_ohlc'] = None

    return cache


def compute_valid_indices(cache: Dict, prediction_horizon: int = 24) -> np.ndarray:
    """Compute valid sample indices (same logic as HierarchicalDataset)."""
    seq_lengths = cache['tf_meta']['sequence_lengths']
    max_seq_len = max(seq_lengths.values())

    n_5min_bars = cache['tf_5min_shape'][0]

    # Valid indices in 5min array
    min_context = max_seq_len
    horizon_5min = prediction_horizon // 5 + 1

    valid_start = min_context
    valid_end = n_5min_bars - horizon_5min

    valid_indices = np.arange(valid_start, valid_end)
    print(f"  Valid sample indices: {len(valid_indices):,} (from {valid_start} to {valid_end})")

    return valid_indices


def precompute_breakout_labels(
    cache: Dict,
    valid_indices: np.ndarray,
    prediction_horizon: int = 24
) -> Dict[str, np.ndarray]:
    """
    Pre-compute breakout labels for all valid samples.

    Fix #1: Eliminates linear regression computation in __getitem__.
    """
    print("\nPre-computing breakout labels (Fix #1)...")

    n_samples = len(valid_indices)

    # Pre-allocate arrays
    breakout_labels = {
        'breakout_occurred': np.zeros(n_samples, dtype=np.float32),
        'breakout_direction': np.full(n_samples, 0.5, dtype=np.float32),
        'breakout_bars_log': np.zeros(n_samples, dtype=np.float32),
        'breakout_magnitude': np.zeros(n_samples, dtype=np.float32),
    }

    tf_5min = cache['tf_5min']
    close_idx = cache['close_idx']
    ts_5min = cache['ts_5min']
    raw_ohlc = cache.get('raw_ohlc')
    raw_ohlc_timestamps = cache.get('raw_ohlc_timestamps')

    start_time = time.time()
    report_interval = n_samples // 20  # Report every 5%

    for i, data_idx in enumerate(valid_indices):
        # Progress reporting
        if i > 0 and i % report_interval == 0:
            elapsed = time.time() - start_time
            pct = i / n_samples * 100
            eta = elapsed / i * (n_samples - i)
            print(f"  {pct:.0f}% complete ({i:,}/{n_samples:,}) - ETA: {eta:.0f}s")

        # Get past prices (60 bars before current)
        past_start = max(0, data_idx - 60)
        past_prices = tf_5min[past_start:data_idx, close_idx]

        # Get current price
        current_price = tf_5min[data_idx - 1, close_idx]

        # Get future prices
        if raw_ohlc is not None and raw_ohlc_timestamps is not None:
            # Use raw 1-min OHLC for more accurate future prices
            ts_5min_val = ts_5min[data_idx]
            approx_1min_idx = np.searchsorted(raw_ohlc_timestamps, ts_5min_val, side='right') - 1
            approx_1min_idx = max(0, min(approx_1min_idx, len(raw_ohlc) - 1))

            future_start = approx_1min_idx
            future_end = min(approx_1min_idx + prediction_horizon, len(raw_ohlc))

            if future_end > future_start:
                future_prices = raw_ohlc[future_start:future_end, 3]  # Close column
            else:
                future_prices = np.array([current_price])
        else:
            # Fallback: use 5min closes
            horizon_5min = prediction_horizon // 5 + 1
            future_end = min(data_idx + horizon_5min, len(tf_5min))
            future_prices = tf_5min[data_idx:future_end, close_idx]

        # Compute breakout label
        label = compute_channel_breakout(
            past_prices=past_prices,
            future_prices=future_prices,
            current_price=current_price
        )

        # Store in arrays
        breakout_labels['breakout_occurred'][i] = label['breakout_occurred']
        breakout_labels['breakout_direction'][i] = label['breakout_direction']
        breakout_labels['breakout_bars_log'][i] = label['breakout_bars_log']
        breakout_labels['breakout_magnitude'][i] = label['breakout_magnitude']

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s ({n_samples / elapsed:.0f} samples/sec)")

    return breakout_labels


def precompute_target_arrays(
    cache: Dict,
    valid_indices: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Pre-compute sample-indexed target arrays for continuation and transition labels.

    Fix #3 (Option B): Dict of arrays indexed by sample.
    Eliminates 2,223 dict insertions per sample in __getitem__.
    """
    print("\nPre-computing target arrays (Fix #3)...")

    n_samples = len(valid_indices)
    ts_5min = cache['ts_5min']

    # Define all target keys and pre-allocate arrays
    target_arrays = {}

    # Base targets (computed elsewhere, but include placeholders)
    base_keys = ['high', 'low', 'hit_band', 'hit_target', 'expected_return', 'overshoot',
                 'continuation_duration', 'continuation_gain', 'continuation_confidence',
                 'price_change_pct', 'horizon_bars_log', 'adaptive_confidence']
    for key in base_keys:
        if key in ['continuation_confidence', 'adaptive_confidence']:
            target_arrays[key] = np.full(n_samples, 0.5, dtype=np.float32)
        else:
            target_arrays[key] = np.zeros(n_samples, dtype=np.float32)

    # Continuation labels per TF per window
    windows = config.CHANNEL_WINDOW_SIZES  # [100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10]
    cont_fields = ['duration', 'hit_upper', 'hit_midline', 'hit_lower', 'confidence', 'valid']

    for tf in HIERARCHICAL_TIMEFRAMES:
        target_arrays[f'cont_{tf}_gain'] = np.zeros(n_samples, dtype=np.float32)
        target_arrays[f'cont_{tf}_valid'] = np.zeros(n_samples, dtype=np.float32)

        for window in windows:
            for field in cont_fields:
                key = f'cont_{tf}_w{window}_{field}'
                if field == 'confidence':
                    target_arrays[key] = np.full(n_samples, 0.5, dtype=np.float32)
                else:
                    target_arrays[key] = np.zeros(n_samples, dtype=np.float32)

    # Transition labels per TF
    trans_fields = ['type', 'switch_to', 'direction', 'slope', 'valid']
    for tf in HIERARCHICAL_TIMEFRAMES:
        for field in trans_fields:
            key = f'trans_{tf}_{field}'
            if field == 'direction':
                target_arrays[key] = np.ones(n_samples, dtype=np.float32)  # Default BEAR
            elif field == 'valid':
                target_arrays[key] = np.zeros(n_samples, dtype=np.float32)
            else:
                target_arrays[key] = np.zeros(n_samples, dtype=np.float32)

    print(f"  Pre-allocated {len(target_arrays)} target arrays")

    # Fill continuation labels
    print("  Filling continuation labels...")
    start_time = time.time()
    report_interval = n_samples // 20

    for i, data_idx in enumerate(valid_indices):
        if i > 0 and i % report_interval == 0:
            elapsed = time.time() - start_time
            pct = i / n_samples * 100
            eta = elapsed / i * (n_samples - i)
            print(f"    {pct:.0f}% complete ({i:,}/{n_samples:,}) - ETA: {eta:.0f}s")

        # Get timestamp for this sample
        ts_val = int(ts_5min[data_idx])

        # Fill continuation labels for each TF
        for tf in HIERARCHICAL_TIMEFRAMES:
            if tf not in cache['cont_labels'] or tf not in cache['cont_ts_to_idx']:
                continue

            cont_df = cache['cont_labels'][tf]
            ts_to_idx = cache['cont_ts_to_idx'][tf]

            row_idx = ts_to_idx.get(ts_val)
            if row_idx is None:
                continue

            # TF-level fields
            if 'max_gain_pct' in cont_df.columns:
                target_arrays[f'cont_{tf}_gain'][i] = float(cont_df['max_gain_pct'].iloc[row_idx])
            target_arrays[f'cont_{tf}_valid'][i] = 1.0

            # Window-level fields
            for window in windows:
                valid_col = f'w{window}_valid'
                if valid_col in cont_df.columns and cont_df[valid_col].iloc[row_idx] > 0:
                    target_arrays[f'cont_{tf}_w{window}_valid'][i] = 1.0

                    for field in ['duration', 'hit_upper', 'hit_midline', 'hit_lower', 'confidence']:
                        col = f'w{window}_{field}'
                        if col in cont_df.columns:
                            target_arrays[f'cont_{tf}_w{window}_{field}'][i] = float(cont_df[col].iloc[row_idx])

        # Fill transition labels for each TF
        for tf in HIERARCHICAL_TIMEFRAMES:
            if tf not in cache['trans_labels'] or tf not in cache['trans_ts_to_idx']:
                continue

            trans_df = cache['trans_labels'][tf]
            ts_to_idx = cache['trans_ts_to_idx'][tf]

            row_idx = ts_to_idx.get(ts_val)
            if row_idx is None:
                continue

            target_arrays[f'trans_{tf}_valid'][i] = 1.0

            if 'transition_type' in trans_df.columns:
                target_arrays[f'trans_{tf}_type'][i] = float(trans_df['transition_type'].iloc[row_idx])
            if 'switch_to_tf' in trans_df.columns:
                val = trans_df['switch_to_tf'].iloc[row_idx]
                if val is not None and not pd.isna(val):
                    target_arrays[f'trans_{tf}_switch_to'][i] = float(HIERARCHICAL_TIMEFRAMES.index(val)) if val in HIERARCHICAL_TIMEFRAMES else 0.0
            if 'new_direction' in trans_df.columns:
                target_arrays[f'trans_{tf}_direction'][i] = float(trans_df['new_direction'].iloc[row_idx])
            if 'new_slope' in trans_df.columns:
                target_arrays[f'trans_{tf}_slope'][i] = float(trans_df['new_slope'].iloc[row_idx])

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s ({n_samples / elapsed:.0f} samples/sec)")

    return target_arrays


def save_precomputed(
    cache_dir: Path,
    cache_key: str,
    valid_indices: np.ndarray,
    breakout_labels: Dict[str, np.ndarray],
    target_arrays: Dict[str, np.ndarray]
):
    """Save pre-computed data to cache directory."""
    print("\nSaving pre-computed data...")

    # Save valid indices
    indices_path = cache_dir / f"precomputed_valid_indices_{cache_key}.npy"
    np.save(indices_path, valid_indices)
    print(f"  Saved {indices_path.name} ({len(valid_indices):,} indices)")

    # v5.9.6: Save as individual .npy files for mmap sharing across workers
    # Create directories for breakout and targets
    breakout_dir = cache_dir / f"precomputed_breakout_{cache_key}.mmap"
    targets_dir = cache_dir / f"precomputed_targets_{cache_key}.mmap"
    breakout_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)

    # Save breakout labels as individual .npy files
    breakout_size = 0
    for key, arr in breakout_labels.items():
        np.save(breakout_dir / f'{key}.npy', arr)
        breakout_size += (breakout_dir / f'{key}.npy').stat().st_size
    print(f"  Saved {breakout_dir.name}/ ({breakout_size/1e6:.1f} MB, {len(breakout_labels)} fields)")

    # Save target arrays as individual .npy files
    targets_size = 0
    for key, arr in target_arrays.items():
        np.save(targets_dir / f'{key}.npy', arr)
        targets_size += (targets_dir / f'{key}.npy').stat().st_size
    print(f"  Saved {targets_dir.name}/ ({targets_size/1e6:.1f} MB, {len(target_arrays)} fields)")

    # Save metadata
    meta = {
        'version': '5.9.6',
        'cache_key': cache_key,
        'n_samples': len(valid_indices),
        'n_breakout_fields': len(breakout_labels),
        'n_target_fields': len(target_arrays),
        'breakout_keys': list(breakout_labels.keys()),
        'target_keys': list(target_arrays.keys()),
        'format': 'mmap_npy',  # v5.9.6: Individual .npy files for mmap
    }
    meta_path = cache_dir / f"precomputed_meta_{cache_key}.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved {meta_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute targets for fast training")
    parser.add_argument('--cache-dir', type=str, default='data/feature_cache',
                        help='Path to feature cache directory')
    parser.add_argument('--prediction-horizon', type=int, default=24,
                        help='Prediction horizon in 1-min bars')
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f"ERROR: Cache directory not found: {cache_dir}")
        sys.exit(1)

    print("=" * 70)
    print("Pre-compute Targets for Fast Training (v5.9.4)")
    print("=" * 70)
    print(f"Cache directory: {cache_dir}")
    print(f"Prediction horizon: {args.prediction_horizon} bars")
    print()

    # Load existing cache
    cache = load_existing_cache(cache_dir)

    # Compute valid indices
    valid_indices = compute_valid_indices(cache, args.prediction_horizon)

    # Pre-compute breakout labels (Fix #1)
    breakout_labels = precompute_breakout_labels(
        cache, valid_indices, args.prediction_horizon
    )

    # Pre-compute target arrays (Fix #3)
    target_arrays = precompute_target_arrays(cache, valid_indices)

    # Save results
    save_precomputed(
        cache_dir,
        cache['cache_key'],
        valid_indices,
        breakout_labels,
        target_arrays
    )

    print()
    print("=" * 70)
    print("Pre-computation complete!")
    print("=" * 70)
    print()
    print("To use pre-computed data, the HierarchicalDataset will automatically")
    print("detect and load these files when they exist in the cache directory.")
    print()
    print("Expected speedup:")
    print("  - Fix #1 (breakout labels): ~3-5 min/epoch saved")
    print("  - Fix #3 (target arrays): ~7-12 min/epoch saved")
    print("  - Combined: ~10-17 min/epoch saved")


if __name__ == '__main__':
    main()
