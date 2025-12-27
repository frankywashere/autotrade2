"""
Validate that precomputed targets match original per-sample computation.

Usage:
    python scripts/validate_precomputed.py

This script:
1. Loads the precomputed breakout labels
2. Recalculates breakout for a sample of indices using original method
3. Compares results and reports any differences
"""

import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path

# Add parent to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import config


def compute_channel_breakout_original(
    past_prices: np.ndarray,
    future_prices: np.ndarray,
    current_price: float,
    lookback: int = 60,
    channel_std: float = 2.0,
    breakout_threshold: float = 1.0
) -> dict:
    """
    Original method from hierarchical_dataset.py (lines 2068-2159).
    """
    result = {
        'breakout_occurred': 0.0,
        'breakout_direction': 0.5,
        'breakout_bars_log': np.log(len(future_prices) + 1),
        'breakout_magnitude': 0.0
    }

    if len(past_prices) < lookback or len(past_prices) < 10:
        return result

    y = past_prices[-lookback:]
    X = np.arange(lookback)

    X_mean = X.mean()
    y_mean = y.mean()
    slope = np.sum((X - X_mean) * (y - y_mean)) / (np.sum((X - X_mean) ** 2) + 1e-10)
    intercept = y_mean - slope * X_mean

    fitted = slope * X + intercept
    residuals = y - fitted
    channel_width = np.std(residuals) + 1e-10  # Original adds epsilon inline

    upper_bounds = []
    lower_bounds = []
    for i in range(len(future_prices)):
        future_bar_idx = lookback + i
        projected_center = slope * future_bar_idx + intercept
        upper_bounds.append(projected_center + channel_std * channel_width)
        lower_bounds.append(projected_center - channel_std * channel_width)

    upper_bounds = np.array(upper_bounds)
    lower_bounds = np.array(lower_bounds)

    breakout_threshold_dist = breakout_threshold * channel_width

    for i, price in enumerate(future_prices):
        if price > upper_bounds[i] + breakout_threshold_dist:
            result['breakout_occurred'] = 1.0
            result['breakout_direction'] = 1.0
            result['breakout_bars_log'] = np.log(i + 1 + 1e-6)
            result['breakout_magnitude'] = (price - upper_bounds[i]) / channel_width
            break
        elif price < lower_bounds[i] - breakout_threshold_dist:
            result['breakout_occurred'] = 1.0
            result['breakout_direction'] = 0.0
            result['breakout_bars_log'] = np.log(i + 1 + 1e-6)
            result['breakout_magnitude'] = (lower_bounds[i] - price) / channel_width
            break

    return result


def compute_channel_breakout_precompute(
    past_prices: np.ndarray,
    future_prices: np.ndarray,
    current_price: float,
    lookback: int = 60,
    channel_std: float = 2.0,
    breakout_threshold: float = 1.0
) -> dict:
    """
    Precompute method from precompute_targets.py (lines 36-117).
    """
    result = {
        'breakout_occurred': 0.0,
        'breakout_direction': 0.5,
        'breakout_bars_log': np.log(len(future_prices) + 1) if len(future_prices) > 0 else 0.0,
        'breakout_magnitude': 0.0
    }

    if len(past_prices) < lookback or len(past_prices) < 10:
        return result

    if len(future_prices) == 0:
        return result

    y = past_prices[-lookback:]
    X = np.arange(lookback)

    X_mean = X.mean()
    y_mean = y.mean()
    denominator = np.sum((X - X_mean) ** 2)
    if denominator < 1e-10:
        return result

    slope = np.sum((X - X_mean) * (y - y_mean)) / (denominator + 1e-10)
    intercept = y_mean - slope * X_mean

    fitted = slope * X + intercept
    residuals = y - fitted
    channel_width = np.std(residuals)
    if channel_width < 1e-10:  # Precompute returns early
        return result

    upper_bounds = []
    lower_bounds = []
    for i in range(len(future_prices)):
        future_bar_idx = lookback + i
        projected_center = slope * future_bar_idx + intercept
        upper_bounds.append(projected_center + channel_std * channel_width)
        lower_bounds.append(projected_center - channel_std * channel_width)

    upper_bounds = np.array(upper_bounds)
    lower_bounds = np.array(lower_bounds)

    breakout_threshold_dist = breakout_threshold * channel_width

    for i, price in enumerate(future_prices):
        if price > upper_bounds[i] + breakout_threshold_dist:
            result['breakout_occurred'] = 1.0
            result['breakout_direction'] = 1.0
            result['breakout_bars_log'] = np.log(i + 1 + 1e-6)
            result['breakout_magnitude'] = (price - upper_bounds[i]) / channel_width
            break
        elif price < lower_bounds[i] - breakout_threshold_dist:
            result['breakout_occurred'] = 1.0
            result['breakout_direction'] = 0.0
            result['breakout_bars_log'] = np.log(i + 1 + 1e-6)
            result['breakout_magnitude'] = (lower_bounds[i] - price) / channel_width
            break

    return result


def main():
    cache_dir = Path("data/feature_cache")

    print("=" * 70)
    print("Validate Precomputed Targets")
    print("=" * 70)

    # Find cache key from tf_meta
    tf_meta_files = sorted(cache_dir.glob("tf_meta_*.json"))
    if not tf_meta_files:
        print("ERROR: No tf_meta_*.json found")
        return

    with open(tf_meta_files[-1]) as f:
        tf_meta = json.load(f)
    cache_key = tf_meta['cache_key']
    print(f"Cache key: {cache_key}")

    # Load precomputed breakout
    breakout_path = cache_dir / f"precomputed_breakout_{cache_key}.npz"
    if not breakout_path.exists():
        print(f"ERROR: Precomputed breakout not found: {breakout_path}")
        return

    precomputed = dict(np.load(breakout_path))
    n_samples = len(precomputed['breakout_occurred'])
    print(f"Loaded precomputed breakout: {n_samples:,} samples")

    # Load valid indices
    indices_path = cache_dir / f"precomputed_valid_indices_{cache_key}.npy"
    valid_indices = np.load(indices_path)
    print(f"Loaded valid indices: {len(valid_indices):,}")

    # Load 5min sequence
    seq_path = cache_dir / f"tf_sequence_5min_{cache_key}.npy"
    tf_5min = np.load(str(seq_path), mmap_mode='r')
    print(f"Loaded 5min sequence: {tf_5min.shape}")

    # Load timestamps
    ts_path = cache_dir / f"tf_timestamps_5min_{cache_key}.npy"
    ts_5min = np.load(ts_path)

    # Find close column
    columns_5min = tf_meta['timeframe_columns']['5min']
    close_idx = None
    for i, col in enumerate(columns_5min):
        if col == 'tsla_close' or 'close' in col.lower():
            close_idx = i
            break
    print(f"Close column index: {close_idx}")

    # Load raw OHLC for future prices
    raw_ohlc_path = Path("data/tsla_1min_data.csv")
    if raw_ohlc_path.exists():
        raw_df = pd.read_csv(raw_ohlc_path, parse_dates=['timestamp'], index_col='timestamp')
        raw_ohlc = raw_df[['tsla_open', 'tsla_high', 'tsla_low', 'tsla_close']].values
        raw_ohlc_timestamps = raw_df.index.values.astype('datetime64[ns]').astype('int64')
        print(f"Loaded raw OHLC: {len(raw_ohlc):,} bars")
    else:
        print("WARNING: Raw OHLC not found, using 5min closes")
        raw_ohlc = None
        raw_ohlc_timestamps = None

    # Sample indices to validate
    np.random.seed(42)
    sample_size = min(1000, n_samples)
    sample_indices = np.random.choice(n_samples, sample_size, replace=False)
    sample_indices.sort()

    print(f"\nValidating {sample_size} random samples...")
    print("-" * 70)

    prediction_horizon = 24
    mismatches = []
    edge_cases = []

    for i, sample_idx in enumerate(sample_indices):
        if i > 0 and i % 200 == 0:
            print(f"  Checked {i}/{sample_size}...")

        data_idx = valid_indices[sample_idx]

        # Get past prices
        past_start = max(0, data_idx - 60)
        past_prices = tf_5min[past_start:data_idx, close_idx].copy()

        # Get current price
        current_price = tf_5min[data_idx - 1, close_idx]

        # Get future prices
        if raw_ohlc is not None:
            ts_5min_val = ts_5min[data_idx]
            approx_1min_idx = np.searchsorted(raw_ohlc_timestamps, ts_5min_val, side='right') - 1
            approx_1min_idx = max(0, min(approx_1min_idx, len(raw_ohlc) - 1))

            future_start = approx_1min_idx
            future_end = min(approx_1min_idx + prediction_horizon, len(raw_ohlc))

            if future_end > future_start:
                future_prices = raw_ohlc[future_start:future_end, 3].copy()
            else:
                future_prices = np.array([current_price])
        else:
            horizon_5min = prediction_horizon // 5 + 1
            future_end = min(data_idx + horizon_5min, len(tf_5min))
            future_prices = tf_5min[data_idx:future_end, close_idx].copy()

        # Compute using both methods
        result_original = compute_channel_breakout_original(
            past_prices, future_prices, current_price
        )
        result_precompute = compute_channel_breakout_precompute(
            past_prices, future_prices, current_price
        )

        # Get stored precomputed values
        stored = {
            'breakout_occurred': precomputed['breakout_occurred'][sample_idx],
            'breakout_direction': precomputed['breakout_direction'][sample_idx],
            'breakout_bars_log': precomputed['breakout_bars_log'][sample_idx],
            'breakout_magnitude': precomputed['breakout_magnitude'][sample_idx],
        }

        # Compare original vs precompute method
        method_match = True
        for key in result_original:
            if abs(result_original[key] - result_precompute[key]) > 1e-6:
                method_match = False
                break

        # Compare precompute method vs stored (use float32 tolerance since stored as float32)
        stored_match = True
        for key in stored:
            if abs(result_precompute[key] - stored[key]) > 1e-4:  # float32 tolerance
                stored_match = False
                break

        # Check for edge case (channel_width near zero)
        y = past_prices[-60:] if len(past_prices) >= 60 else past_prices
        if len(y) >= 10:
            X = np.arange(len(y))
            X_mean, y_mean = X.mean(), y.mean()
            slope = np.sum((X - X_mean) * (y - y_mean)) / (np.sum((X - X_mean) ** 2) + 1e-10)
            intercept = y_mean - slope * X_mean
            residuals = y - (slope * X + intercept)
            channel_width = np.std(residuals)

            if channel_width < 1e-8:
                edge_cases.append({
                    'sample_idx': sample_idx,
                    'channel_width': channel_width,
                    'original': result_original,
                    'precompute': result_precompute,
                    'stored': stored
                })

        if not method_match or not stored_match:
            mismatches.append({
                'sample_idx': sample_idx,
                'data_idx': data_idx,
                'method_match': method_match,
                'stored_match': stored_match,
                'original': result_original,
                'precompute': result_precompute,
                'stored': stored
            })

    # Report results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nSamples checked: {sample_size}")
    print(f"Mismatches: {len(mismatches)}")
    print(f"Edge cases (channel_width < 1e-8): {len(edge_cases)}")

    if mismatches:
        print(f"\n⚠️  Found {len(mismatches)} mismatches:")
        for m in mismatches[:5]:  # Show first 5
            print(f"\n  Sample {m['sample_idx']}:")
            print(f"    Method match: {m['method_match']}, Stored match: {m['stored_match']}")
            print(f"    Original:   {m['original']}")
            print(f"    Precompute: {m['precompute']}")
            print(f"    Stored:     {m['stored']}")
    else:
        print("\n✓ All samples match!")

    if edge_cases:
        print(f"\n📊 Edge cases (channel_width ≈ 0):")
        for e in edge_cases[:3]:  # Show first 3
            print(f"\n  Sample {e['sample_idx']}, channel_width={e['channel_width']:.2e}:")
            print(f"    Original:   {e['original']}")
            print(f"    Precompute: {e['precompute']}")
            print(f"    Stored:     {e['stored']}")

    # Summary
    match_rate = (sample_size - len(mismatches)) / sample_size * 100
    print(f"\n{'=' * 70}")
    print(f"Match rate: {match_rate:.2f}%")
    if match_rate == 100.0:
        print("✓ Precomputed values are identical to original calculation")
    elif match_rate > 99.9:
        print("✓ Precomputed values match except for rare edge cases")
    else:
        print("⚠️  Significant differences found - investigate mismatches")
    print("=" * 70)


if __name__ == '__main__':
    main()
