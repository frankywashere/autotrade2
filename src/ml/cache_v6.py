"""
v6.0 Unified Cache Format and Label Generation

This module handles:
1. Break detection with return tracking
2. Transition label generation
3. Unified cache format (.npz files per timeframe)
4. Cache metadata generation

Cache Structure:
data/feature_cache_v6/
├── tf_5min_v6.0.0.npz      # 5-min TF: features + ohlc + labels
├── tf_15min_v6.0.0.npz     # etc.
├── ...
├── vix_v6.0.0.npz          # VIX sequences
└── cache_meta_v6.0.0.json  # Metadata
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from tqdm import tqdm
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

import config as project_config


# =============================================================================
# CONSTANTS
# =============================================================================

VERSION = "6.0.0"

TIMEFRAMES = ['5min', '15min', '30min', '1h', '2h', '3h', '4h',
              'daily', 'weekly', 'monthly', '3month']

# 14 window sizes for channel analysis
WINDOWS = [100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10]

# Transition type constants
TRANSITION_CONTINUE = 0   # Same channel extends
TRANSITION_SWITCH_TF = 1  # Different TF's channel takes over
TRANSITION_REVERSE = 2    # Direction reverses
TRANSITION_SIDEWAYS = 3   # Price consolidates

# Direction constants
DIRECTION_BEAR = 0
DIRECTION_BULL = 1
DIRECTION_SIDEWAYS = 2


# =============================================================================
# BREAK DETECTION WITH RETURN TRACKING
# =============================================================================

def detect_break_with_return(
    future_ohlc: np.ndarray,
    slope: float,
    intercept: float,
    residual_std: float,
    window: int,
    max_scan_bars: int = 500,
    return_threshold_bars: int = 3,
) -> Dict[str, Any]:
    """
    Detect channel break AND track if price returns.

    A channel "break" = close outside ±2σ bounds.
    A "return" = price stays inside for return_threshold_bars consecutive bars.

    Args:
        future_ohlc: [max_bars, 4] - Future OHLC (open, high, low, close)
        slope: Channel slope (price per bar)
        intercept: Channel intercept (price at x=0)
        residual_std: Channel width (1σ)
        window: Lookback window size
        max_scan_bars: Maximum bars to scan forward
        return_threshold_bars: Bars inside to count as "returned"

    Returns:
        {
            'first_break_bar': int - Bar of first break (or max_scan_bars if none)
            'break_direction': int - -1=below, 0=none, 1=above
            'returned': bool - Returned after break?
            'bars_to_return': int - Bars until return (0 if no break/return)
            'bars_outside': int - Total bars outside
            'max_consecutive_outside': int - Longest streak
            'final_duration': int - Effective duration (with returns)
            'price_sequence': list[float] - % changes from start
            'hit_upper': bool - Hit upper bound?
            'hit_midline': bool - Hit midline?
            'hit_lower': bool - Hit lower bound?
            'bars_to_upper': int - Bars until hit upper
            'bars_to_midline': int - Bars until hit midline
            'bars_to_lower': int - Bars until hit lower
        }
    """
    if len(future_ohlc) == 0:
        return {
            'first_break_bar': 0,
            'break_direction': 0,
            'returned': False,
            'bars_to_return': 0,
            'bars_outside': 0,
            'max_consecutive_outside': 0,
            'final_duration': 0,
            'price_sequence': [],
            'hit_upper': False,
            'hit_midline': False,
            'hit_lower': False,
            'bars_to_upper': max_scan_bars,
            'bars_to_midline': max_scan_bars,
            'bars_to_lower': max_scan_bars,
        }

    future_closes = future_ohlc[:, 3]  # Close prices
    future_highs = future_ohlc[:, 1]
    future_lows = future_ohlc[:, 2]
    start_price = future_closes[0] if len(future_closes) > 0 else 1.0

    first_break_bar = None
    break_direction = 0
    returned = False
    bars_to_return = 0
    total_bars_outside = 0
    max_consecutive_outside = 0
    current_consecutive_outside = 0
    consecutive_inside_after_break = 0
    price_sequence = []

    # Hit tracking
    hit_upper = False
    hit_midline = False
    hit_lower = False
    bars_to_upper = max_scan_bars
    bars_to_midline = max_scan_bars
    bars_to_lower = max_scan_bars

    scan_length = min(len(future_closes), max_scan_bars)

    for bar_idx in range(scan_length):
        # Calculate % change from start
        price = future_closes[bar_idx]
        high = future_highs[bar_idx]
        low = future_lows[bar_idx]
        pct_change = (price - start_price) / start_price * 100 if start_price > 0 else 0
        price_sequence.append(pct_change)

        # Project channel bounds to this bar
        x_pos = window + bar_idx
        center = slope * x_pos + intercept
        upper = center + (2.0 * residual_std)
        lower = center - (2.0 * residual_std)

        # Check for hits (use high/low for touch detection)
        if not hit_upper and high >= upper:
            hit_upper = True
            bars_to_upper = bar_idx

        if not hit_midline:
            # Midline hit if price crosses center
            if low <= center <= high:
                hit_midline = True
                bars_to_midline = bar_idx

        if not hit_lower and low <= lower:
            hit_lower = True
            bars_to_lower = bar_idx

        # Check for break (use close for break detection)
        is_outside = (price > upper) or (price < lower)

        if is_outside:
            total_bars_outside += 1
            current_consecutive_outside += 1
            consecutive_inside_after_break = 0
            max_consecutive_outside = max(max_consecutive_outside, current_consecutive_outside)

            if first_break_bar is None:
                first_break_bar = bar_idx
                break_direction = 1 if price > upper else -1
        else:
            current_consecutive_outside = 0

            if first_break_bar is not None and not returned:
                consecutive_inside_after_break += 1

                if consecutive_inside_after_break >= return_threshold_bars:
                    returned = True
                    bars_to_return = bar_idx - first_break_bar

    # Calculate final duration
    if first_break_bar is None:
        final_duration = scan_length
    elif returned:
        final_duration = scan_length  # Channel effectively still valid
    else:
        final_duration = first_break_bar

    return {
        'first_break_bar': first_break_bar if first_break_bar is not None else scan_length,
        'break_direction': break_direction,
        'returned': returned,
        'bars_to_return': bars_to_return,
        'bars_outside': total_bars_outside,
        'max_consecutive_outside': max_consecutive_outside,
        'final_duration': final_duration,
        'price_sequence': price_sequence,
        'hit_upper': hit_upper,
        'hit_midline': hit_midline,
        'hit_lower': hit_lower,
        'bars_to_upper': bars_to_upper,
        'bars_to_midline': bars_to_midline,
        'bars_to_lower': bars_to_lower,
    }


def compute_channel_state(
    ohlc: np.ndarray,
    window: int,
) -> Dict[str, float]:
    """
    Compute channel state (slope, intercept, R², etc.) for a window.

    Args:
        ohlc: [window, 4] - OHLC data for the window
        window: Window size

    Returns:
        {
            'valid': bool,
            'slope': float,
            'intercept': float,
            'residual_std': float,
            'r_squared': float,
            'upper_dist': float,
            'lower_dist': float,
            'position': float,
        }
    """
    if len(ohlc) < window:
        return {
            'valid': False,
            'slope': 0.0,
            'intercept': 0.0,
            'residual_std': 1.0,
            'r_squared': 0.0,
            'upper_dist': 0.0,
            'lower_dist': 0.0,
            'position': 0.5,
        }

    closes = ohlc[-window:, 3]
    current_price = closes[-1]

    # Fit linear regression
    x = np.arange(window)
    x_mean = x.mean()
    y_mean = closes.mean()

    numerator = np.sum((x - x_mean) * (closes - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator < 1e-10:
        slope = 0.0
    else:
        slope = numerator / denominator

    intercept = y_mean - slope * x_mean

    # Calculate R²
    predicted = slope * x + intercept
    ss_res = np.sum((closes - predicted) ** 2)
    ss_tot = np.sum((closes - y_mean) ** 2)

    if ss_tot < 1e-10:
        r_squared = 0.0
    else:
        r_squared = 1 - (ss_res / ss_tot)
        r_squared = max(0, r_squared)

    # Residual standard deviation
    residuals = closes - predicted
    residual_std = np.std(residuals) if len(residuals) > 1 else 1.0
    residual_std = max(residual_std, 1e-6)

    # Current position in channel
    current_pred = slope * (window - 1) + intercept
    upper = current_pred + 2 * residual_std
    lower = current_pred - 2 * residual_std

    # Distances as percentage
    upper_dist = (upper - current_price) / current_price * 100 if current_price > 0 else 0
    lower_dist = (current_price - lower) / current_price * 100 if current_price > 0 else 0

    # Position in channel (0=lower, 0.5=center, 1=upper)
    channel_range = upper - lower
    if channel_range > 1e-6:
        position = (current_price - lower) / channel_range
        position = np.clip(position, 0, 1)
    else:
        position = 0.5

    # Valid if R² is reasonable
    valid = r_squared > 0.3

    return {
        'valid': valid,
        'slope': slope,
        'intercept': intercept,
        'residual_std': residual_std,
        'r_squared': r_squared,
        'upper_dist': upper_dist,
        'lower_dist': lower_dist,
        'position': position,
    }


# =============================================================================
# TRANSITION LABEL GENERATION
# =============================================================================

def detect_transition(
    future_ohlc: np.ndarray,
    current_tf: str,
    current_direction: int,
    all_tf_durations: Dict[str, int],
    break_bar: int,
) -> Dict[str, int]:
    """
    Determine what happens after channel breaks.

    Args:
        future_ohlc: [max_bars, 4] - Future OHLC data
        current_tf: Current timeframe name
        current_direction: Current channel direction (0=bear, 1=bull, 2=sideways)
        all_tf_durations: Dict mapping TF name → final duration
        break_bar: Bar when current channel broke

    Returns:
        {
            'transition_type': int (0-3),
            'direction': int (0-2),
            'next_tf': int (0-10),
        }
    """
    if break_bar >= len(future_ohlc) - 10:
        # Not enough data after break
        return {'transition_type': 0, 'direction': DIRECTION_SIDEWAYS, 'next_tf': 0}

    # Analyze price action after break
    post_break = future_ohlc[break_bar:break_bar + 20]
    if len(post_break) < 5:
        return {'transition_type': 0, 'direction': DIRECTION_SIDEWAYS, 'next_tf': 0}

    # Determine direction from returns
    total_return = (post_break[-1, 3] - post_break[0, 3]) / post_break[0, 3]
    if total_return > 0.01:
        direction = DIRECTION_BULL
    elif total_return < -0.01:
        direction = DIRECTION_BEAR
    else:
        direction = DIRECTION_SIDEWAYS

    # Check if another TF takes over
    current_tf_idx = TIMEFRAMES.index(current_tf) if current_tf in TIMEFRAMES else 0
    best_other_tf = None
    best_other_duration = 0

    for tf, dur in all_tf_durations.items():
        if tf == current_tf:
            continue
        if dur > best_other_duration:
            best_other_duration = dur
            if tf in TIMEFRAMES:
                best_other_tf = TIMEFRAMES.index(tf)

    # Determine transition type
    if best_other_tf is not None and best_other_duration > 20:
        transition_type = TRANSITION_SWITCH_TF
        next_tf = best_other_tf
    elif direction == DIRECTION_SIDEWAYS:
        transition_type = TRANSITION_SIDEWAYS
        next_tf = current_tf_idx
    elif (direction == DIRECTION_BULL and current_direction == DIRECTION_BEAR) or \
         (direction == DIRECTION_BEAR and current_direction == DIRECTION_BULL):
        transition_type = TRANSITION_REVERSE
        next_tf = current_tf_idx
    else:
        transition_type = TRANSITION_CONTINUE
        next_tf = current_tf_idx

    return {
        'transition_type': transition_type,
        'direction': direction,
        'next_tf': next_tf,
    }


# =============================================================================
# CACHE GENERATION
# =============================================================================

def generate_v6_cache(
    features_df: pd.DataFrame,
    raw_ohlc_df: pd.DataFrame,
    output_dir: str = "data/feature_cache_v6",
    v5_cache_dir: str = None,
    max_scan_bars: int = 500,
    return_threshold_bars: int = 3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Generate v6.0 unified cache with all labels.

    This is the main entry point for cache generation.

    Args:
        features_df: DataFrame with computed features (indexed by timestamp)
        raw_ohlc_df: Raw OHLC DataFrame (indexed by timestamp)
        output_dir: Output directory for cache files
        v5_cache_dir: Path to v5.9 cache directory (to load features from .npy files)
        max_scan_bars: Maximum bars to scan forward for breaks
        return_threshold_bars: Bars inside to count as "returned"
        verbose: Print progress

    Returns:
        Metadata dict
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"v6.0 Cache Generation")
        print(f"{'='*60}")
        print(f"Output: {output_path}")
        print(f"Features: {len(features_df):,} rows × {len(features_df.columns)} cols")
        print(f"OHLC: {len(raw_ohlc_df):,} rows")
        if v5_cache_dir:
            print(f"v5.9 cache: {v5_cache_dir}")

    # Extract OHLC array
    ohlc_cols = ['tsla_open', 'tsla_high', 'tsla_low', 'tsla_close']
    if all(c in raw_ohlc_df.columns for c in ohlc_cols):
        ohlc_array = raw_ohlc_df[ohlc_cols].values
    else:
        raise ValueError(f"Missing OHLC columns. Expected: {ohlc_cols}")

    metadata = {
        'version': VERSION,
        'created': datetime.now().isoformat(),
        'data_range': {
            'start': str(features_df.index[0]),
            'end': str(features_df.index[-1]),
        },
        'source_rows': len(features_df),
        'windows': WINDOWS,
        'break_detection': {
            'method': '2sigma_with_return_tracking',
            'return_threshold_bars': return_threshold_bars,
            'max_scan_bars': max_scan_bars,
        },
        'timeframes': {},
    }

    # Process each timeframe
    for tf in TIMEFRAMES:
        if verbose:
            print(f"\n  Processing {tf}...")

        # FIX #3: Load v5.9 features if v5_cache_dir provided
        features_array = None
        if v5_cache_dir:
            v5_path = Path(v5_cache_dir)
            # Find v5.9 features file for this TF (glob pattern to handle version suffix)
            pattern = f"tf_sequence_{tf}_v5.9*.npy"
            matches = list(v5_path.glob(pattern))
            if matches:
                features_path = matches[0]
                if verbose:
                    print(f"    Loading v5.9 features from {features_path.name}...")
                features_array = np.load(str(features_path), mmap_mode='r')
                if verbose:
                    print(f"    ✓ Loaded {features_array.shape[0]:,} bars × {features_array.shape[1]} features")
            else:
                if verbose:
                    print(f"    ⚠️  No v5.9 features found for {tf} (pattern: {pattern})")

        tf_labels = generate_tf_labels(
            features_df=features_df,
            ohlc_array=ohlc_array,
            tf=tf,
            features_array=features_array,
            max_scan_bars=max_scan_bars,
            return_threshold_bars=return_threshold_bars,
            verbose=verbose,
        )

        # Save as .npz
        npz_path = output_path / f"tf_{tf}_{VERSION}.npz"
        np.savez_compressed(str(npz_path), **tf_labels)

        file_size_mb = npz_path.stat().st_size / 1e6
        metadata['timeframes'][tf] = {
            'bars': tf_labels['timestamps'].shape[0] if 'timestamps' in tf_labels else 0,
            'file_size_mb': round(file_size_mb, 2),
        }

        if verbose:
            print(f"    ✓ Saved {npz_path.name} ({file_size_mb:.1f} MB)")

    # Save metadata
    meta_path = output_path / f"cache_meta_{VERSION}.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"\n✓ Cache generation complete!")
        print(f"  Metadata: {meta_path}")

    return metadata


def generate_tf_labels(
    features_df: pd.DataFrame,
    ohlc_array: np.ndarray,
    tf: str,
    features_array: np.ndarray = None,
    max_scan_bars: int = 500,
    return_threshold_bars: int = 3,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Generate all labels for one timeframe.

    Args:
        features_df: Features DataFrame
        ohlc_array: [N, 4] OHLC array
        tf: Timeframe name
        features_array: [N, 1049] Features array (optional, loaded from v5.9 cache)
        max_scan_bars: Maximum bars to scan forward
        return_threshold_bars: Bars inside to count as "returned"
        verbose: Print progress

    Returns:
        Dict of numpy arrays for the .npz file
    """
    n_samples = len(features_df)
    n_windows = len(WINDOWS)

    # Initialize unified cache with OHLC + features + labels
    labels = {
        'timestamps': features_df.index.values.astype('datetime64[ns]').astype('int64'),
        # FIX #2: Add OHLC to unified cache
        'ohlc': ohlc_array.astype(np.float32),
    }

    # FIX #3: Add features to unified cache if provided
    if features_array is not None:
        labels['features'] = features_array.astype(np.float32)

    # Per-window labels
    for w_idx, window in enumerate(WINDOWS):
        prefix = f'w{window}'
        labels[f'{prefix}_valid'] = np.zeros(n_samples, dtype=np.int8)
        labels[f'{prefix}_r_squared'] = np.zeros(n_samples, dtype=np.float32)
        labels[f'{prefix}_slope'] = np.zeros(n_samples, dtype=np.float32)
        labels[f'{prefix}_width'] = np.zeros(n_samples, dtype=np.float32)
        labels[f'{prefix}_first_break_bar'] = np.zeros(n_samples, dtype=np.float32)
        labels[f'{prefix}_final_duration'] = np.zeros(n_samples, dtype=np.float32)
        labels[f'{prefix}_break_direction'] = np.zeros(n_samples, dtype=np.int8)
        labels[f'{prefix}_returned'] = np.zeros(n_samples, dtype=np.int8)
        labels[f'{prefix}_bars_to_return'] = np.zeros(n_samples, dtype=np.float32)
        labels[f'{prefix}_bars_outside'] = np.zeros(n_samples, dtype=np.float32)
        labels[f'{prefix}_max_consecutive_outside'] = np.zeros(n_samples, dtype=np.int8)
        labels[f'{prefix}_hit_upper'] = np.zeros(n_samples, dtype=np.int8)
        labels[f'{prefix}_hit_midline'] = np.zeros(n_samples, dtype=np.int8)
        labels[f'{prefix}_hit_lower'] = np.zeros(n_samples, dtype=np.int8)
        labels[f'{prefix}_bars_to_upper'] = np.zeros(n_samples, dtype=np.float32)
        labels[f'{prefix}_bars_to_midline'] = np.zeros(n_samples, dtype=np.float32)
        labels[f'{prefix}_bars_to_lower'] = np.zeros(n_samples, dtype=np.float32)
        # FIX #1: Add price_sequence storage (variable length, use object array)
        labels[f'{prefix}_price_sequence'] = np.empty(n_samples, dtype=object)

    # Transition labels
    labels['transition_type'] = np.zeros(n_samples, dtype=np.int8)
    labels['transition_direction'] = np.zeros(n_samples, dtype=np.int8)
    labels['transition_next_tf'] = np.zeros(n_samples, dtype=np.int8)

    # Process each sample
    iterator = tqdm(range(n_samples), desc=f"    {tf}", disable=not verbose)

    for i in iterator:
        # Get future OHLC for this sample
        future_start = i + 1
        future_end = min(i + 1 + max_scan_bars, n_samples)
        future_ohlc = ohlc_array[future_start:future_end]

        if len(future_ohlc) < 10:
            continue

        # Get current OHLC for channel computation
        lookback_end = i + 1
        lookback_start = max(0, lookback_end - max(WINDOWS))
        current_ohlc = ohlc_array[lookback_start:lookback_end]

        # Track best window for transition detection
        best_window_duration = 0
        all_window_durations = {}

        # Process each window
        for w_idx, window in enumerate(WINDOWS):
            prefix = f'w{window}'

            if len(current_ohlc) < window:
                continue

            # Compute channel state
            channel_state = compute_channel_state(
                ohlc=current_ohlc,
                window=window,
            )

            labels[f'{prefix}_valid'][i] = 1 if channel_state['valid'] else 0
            labels[f'{prefix}_r_squared'][i] = channel_state['r_squared']
            labels[f'{prefix}_slope'][i] = channel_state['slope']
            labels[f'{prefix}_width'][i] = channel_state['residual_std'] * 4  # 4σ width

            if not channel_state['valid']:
                continue

            # Detect break with return tracking
            break_result = detect_break_with_return(
                future_ohlc=future_ohlc,
                slope=channel_state['slope'],
                intercept=channel_state['intercept'],
                residual_std=channel_state['residual_std'],
                window=window,
                max_scan_bars=max_scan_bars,
                return_threshold_bars=return_threshold_bars,
            )

            labels[f'{prefix}_first_break_bar'][i] = break_result['first_break_bar']
            labels[f'{prefix}_final_duration'][i] = break_result['final_duration']
            labels[f'{prefix}_break_direction'][i] = break_result['break_direction']
            labels[f'{prefix}_returned'][i] = 1 if break_result['returned'] else 0
            labels[f'{prefix}_bars_to_return'][i] = break_result['bars_to_return']
            labels[f'{prefix}_bars_outside'][i] = break_result['bars_outside']
            labels[f'{prefix}_max_consecutive_outside'][i] = break_result['max_consecutive_outside']
            labels[f'{prefix}_hit_upper'][i] = 1 if break_result['hit_upper'] else 0
            labels[f'{prefix}_hit_midline'][i] = 1 if break_result['hit_midline'] else 0
            labels[f'{prefix}_hit_lower'][i] = 1 if break_result['hit_lower'] else 0
            labels[f'{prefix}_bars_to_upper'][i] = break_result['bars_to_upper']
            labels[f'{prefix}_bars_to_midline'][i] = break_result['bars_to_midline']
            labels[f'{prefix}_bars_to_lower'][i] = break_result['bars_to_lower']
            # FIX #1: Store price_sequence (critical for containment loss)
            labels[f'{prefix}_price_sequence'][i] = np.array(break_result['price_sequence'], dtype=np.float32)

            # Track for transition detection
            all_window_durations[f'{tf}_w{window}'] = break_result['final_duration']
            if break_result['final_duration'] > best_window_duration:
                best_window_duration = break_result['final_duration']

        # Transition detection (use best window)
        best_window_idx = 0
        for w_idx, window in enumerate(WINDOWS):
            dur = labels[f'w{window}_final_duration'][i]
            if dur > labels[f'w{WINDOWS[best_window_idx]}_final_duration'][i]:
                best_window_idx = w_idx

        best_window = WINDOWS[best_window_idx]
        break_bar = int(labels[f'w{best_window}_first_break_bar'][i])
        current_direction = 1 if labels[f'w{best_window}_slope'][i] > 0 else 0

        transition = detect_transition(
            future_ohlc=future_ohlc,
            current_tf=tf,
            current_direction=current_direction,
            all_tf_durations={tf: best_window_duration},
            break_bar=break_bar,
        )

        labels['transition_type'][i] = transition['transition_type']
        labels['transition_direction'][i] = transition['direction']
        labels['transition_next_tf'][i] = transition['next_tf']

    return labels


# =============================================================================
# CACHE LOADING
# =============================================================================

def load_v6_cache(
    cache_dir: str = "data/feature_cache_v6",
    timeframes: List[str] = None,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    """
    Load v6.0 cache files.

    Args:
        cache_dir: Cache directory
        timeframes: List of timeframes to load (None = all)

    Returns:
        (tf_data, metadata) where:
            tf_data: Dict[tf] → Dict[key] → np.ndarray
            metadata: Dict from cache_meta.json
    """
    cache_path = Path(cache_dir)

    # Find version
    meta_files = list(cache_path.glob("cache_meta_*.json"))
    if not meta_files:
        raise FileNotFoundError(f"No cache metadata found in {cache_dir}")

    meta_path = meta_files[0]  # Use most recent if multiple
    with open(meta_path) as f:
        metadata = json.load(f)

    version = metadata.get('version', VERSION)

    if timeframes is None:
        timeframes = TIMEFRAMES

    tf_data = {}
    for tf in timeframes:
        npz_path = cache_path / f"tf_{tf}_{version}.npz"
        if npz_path.exists():
            tf_data[tf] = dict(np.load(str(npz_path), allow_pickle=True))
        else:
            print(f"Warning: Missing cache file for {tf}")

    return tf_data, metadata


# =============================================================================
# VALIDATION
# =============================================================================

def validate_v6_cache(cache_dir: str = "data/feature_cache_v6") -> bool:
    """
    Validate v6.0 cache integrity.

    Args:
        cache_dir: Cache directory

    Returns:
        True if valid, False otherwise
    """
    try:
        tf_data, metadata = load_v6_cache(cache_dir)

        print(f"\nv6.0 Cache Validation")
        print(f"{'='*40}")
        print(f"Version: {metadata.get('version', 'unknown')}")
        print(f"Created: {metadata.get('created', 'unknown')}")
        print(f"Data range: {metadata.get('data_range', {})}")

        all_valid = True

        for tf in TIMEFRAMES:
            if tf not in tf_data:
                print(f"  ✗ {tf}: MISSING")
                all_valid = False
                continue

            data = tf_data[tf]
            n_samples = len(data.get('timestamps', []))

            # Check required keys
            required_keys = ['timestamps', 'transition_type']
            for window in WINDOWS[:3]:  # Check first 3 windows
                required_keys.extend([
                    f'w{window}_valid',
                    f'w{window}_final_duration',
                    f'w{window}_r_squared',
                ])

            missing = [k for k in required_keys if k not in data]

            if missing:
                print(f"  ✗ {tf}: Missing keys: {missing[:3]}...")
                all_valid = False
            else:
                print(f"  ✓ {tf}: {n_samples:,} samples")

        return all_valid

    except Exception as e:
        print(f"Validation error: {e}")
        return False


if __name__ == "__main__":
    # CLI for cache validation
    import argparse

    parser = argparse.ArgumentParser(description="v6.0 Cache Tools")
    parser.add_argument("command", choices=["validate", "info"],
                       help="Command to run")
    parser.add_argument("--dir", default="data/feature_cache_v6",
                       help="Cache directory")

    args = parser.parse_args()

    if args.command == "validate":
        valid = validate_v6_cache(args.dir)
        exit(0 if valid else 1)
    elif args.command == "info":
        try:
            _, metadata = load_v6_cache(args.dir)
            print(json.dumps(metadata, indent=2))
        except Exception as e:
            print(f"Error: {e}")
            exit(1)
