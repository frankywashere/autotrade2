"""
Two-Pass Channel Scanner for v15.

Uses the efficient two-pass labeling system:
1. PASS 1: Pre-compute all channels across the entire dataset
2. PASS 2: Generate labels for all channels at once
3. SCAN: For each position, detect channels and look up labels from map (O(1))

This is much faster than scanning forward per-sample because channels and labels
are pre-computed once, then looked up instantly.

Module dependencies:
- data.loader for data loading
- data.resampler for partial bars (keeps incomplete bars with metadata)
- features.tf_extractor for feature extraction
- LOUD failures - no silent try/except
- Validates all outputs before storing

Generates samples with tf_features (8,665 features) compatible with the new model.
"""

import argparse
import multiprocessing as mp
import pickle
import platform
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# V15 module imports
from v15.types import ChannelSample, STANDARD_WINDOWS
from v15.config import (
    SCANNER_CONFIG, TOTAL_FEATURES, FEATURE_COUNTS,
    TIMEFRAMES,  # 10-TF list (no 3month)
    SCANNER_LOOKBACK_5MIN,  # Practical limit for scanning lookback
)
from v15.exceptions import (
    DataLoadError, FeatureExtractionError, ValidationError,
    InvalidFeatureError, ResamplingError,
)
from v15.data import load_market_data, resample_with_partial
from v15.features.tf_extractor import extract_all_tf_features
from v15.features.validation import validate_features
from v15.labels import (
    # Two-pass labeling system imports
    detect_all_channels,
    generate_all_labels,
    get_labels_for_position,
    ChannelMap,
    LabeledChannelMap,
)

# V7 channel detection (still needed)
from v7.core.channel import detect_channels_multi_window, select_best_channel


# =============================================================================
# Multiprocessing Safety Guards
# =============================================================================

def _setup_multiprocessing():
    """Configure multiprocessing for cross-platform compatibility."""
    # Use 'spawn' on macOS (fork is unsafe with certain libraries)
    # Use 'fork' on Linux for better performance
    if platform.system() == 'Darwin':
        try:
            mp.set_start_method('spawn', force=False)
        except RuntimeError:
            pass  # Already set
    # Linux defaults to 'fork' which is fine


def _estimate_memory_per_worker(df_rows: int) -> float:
    """
    Estimate memory usage per worker in MB.

    Args:
        df_rows: Number of rows in the DataFrame

    Returns:
        Estimated memory usage in MB
    """
    # Rough estimate: 8 bytes per float * 6 columns * rows * 3 (TSLA+SPY+VIX)
    return (8 * 6 * df_rows * 3) / (1024 * 1024)


def _check_memory_availability(workers: int, df_rows: int):
    """
    Warn if memory might be insufficient for parallel processing.

    Args:
        workers: Number of worker processes
        df_rows: Number of rows in the DataFrame
    """
    try:
        import psutil

        mem_per_worker = _estimate_memory_per_worker(df_rows)
        total_needed = mem_per_worker * workers
        available = psutil.virtual_memory().available / (1024 * 1024)

        if total_needed > available * 0.8:
            print(f"WARNING: Estimated memory usage ({total_needed:.0f}MB) may exceed "
                  f"available memory ({available:.0f}MB). Consider reducing --workers.")
    except ImportError:
        # psutil not installed, skip memory check
        pass


def _convert_df_to_pickle_safe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert DataFrame to pickle-safe format for worker processes.

    Args:
        df: pandas DataFrame with DatetimeIndex

    Returns:
        Dict with numpy arrays and index info
    """
    return {
        'values': df.values.copy(),
        'columns': list(df.columns),
        'index': df.index.values.copy(),
        'index_name': df.index.name,
    }


def _reconstruct_df_from_pickle_safe(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Reconstruct DataFrame from pickle-safe format.

    Args:
        data: Dict from _convert_df_to_pickle_safe

    Returns:
        Reconstructed pandas DataFrame
    """
    df = pd.DataFrame(
        data['values'],
        columns=data['columns'],
        index=pd.DatetimeIndex(data['index'], name=data['index_name'])
    )
    return df


# Global flag for graceful shutdown
_shutdown_requested = False


def _signal_handler(signum, frame):
    """Handle interrupt signals for graceful shutdown."""
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[INTERRUPT] Shutdown requested. Finishing current work...")


def _worker_process_position(args: Tuple) -> Dict[str, Any]:
    """
    Worker function to process a single position.

    This function is designed to be pickle-safe and handles errors gracefully.

    Args:
        args: Tuple of (idx, tsla_data, spy_data, vix_data, config_dict)
              where data dicts are from _convert_df_to_pickle_safe

    Returns:
        Dict with either 'sample' key (success) or 'error' key (failure)
    """
    try:
        idx, tsla_data, spy_data, vix_data, config_dict = args

        # Reconstruct DataFrames from pickle-safe format
        tsla_df = _reconstruct_df_from_pickle_safe(tsla_data)
        spy_df = _reconstruct_df_from_pickle_safe(spy_data)
        vix_df = _reconstruct_df_from_pickle_safe(vix_data)

        # Import here to avoid issues in worker processes
        from v15.features.tf_extractor import extract_all_tf_features
        from v7.core.channel import detect_channels_multi_window, select_best_channel

        # Get efficient slice
        start_idx = max(0, idx - config_dict['lookback'])
        tsla_slice = tsla_df.iloc[start_idx:idx]
        spy_slice = spy_df.iloc[start_idx:idx]
        vix_slice = vix_df.iloc[start_idx:idx]
        offset = idx - start_idx

        # Detect channels
        channels = detect_channels_multi_window(tsla_slice, windows=config_dict['windows'])
        best_channel, best_window = select_best_channel(channels)

        if best_channel is None or not best_channel.valid:
            return {'idx': idx, 'skipped': True, 'error': None}

        timestamp = tsla_df.index[idx - 1]

        # Extract features
        tf_features = extract_all_tf_features(
            tsla_df=tsla_slice,
            spy_df=spy_slice,
            vix_df=vix_slice,
            timestamp=timestamp,
            source_bar_count=idx,
            include_bar_metadata=True
        )

        return {
            'idx': idx,
            'skipped': False,
            'error': None,
            'timestamp': timestamp,
            'tf_features': tf_features,
            'best_window': best_window,
            'offset': offset,
        }

    except Exception as e:
        return {
            'idx': args[0] if args else -1,
            'skipped': False,
            'error': f"{type(e).__name__}: {str(e)}",
            'traceback': traceback.format_exc(),
        }


def _process_position_batch(batch_args: Tuple) -> List[Dict[str, Any]]:
    """
    Worker function to process a batch of positions.

    This function is designed to be pickle-safe and handles errors gracefully.
    It processes multiple positions in a single call to reduce IPC overhead.

    Args:
        batch_args: Tuple of (positions_batch, tsla_data, spy_data, vix_data,
                             labeled_map, timeframes, windows)
                   where data dicts are from _convert_df_to_pickle_safe

    Returns:
        List of result dicts, each with 'sample', 'error', or 'skipped' key
    """
    try:
        (positions_batch, tsla_data, spy_data, vix_data,
         labeled_map, timeframes, windows) = batch_args

        # Reconstruct DataFrames ONCE for the entire batch
        tsla_df = _reconstruct_df_from_pickle_safe(tsla_data)
        spy_df = _reconstruct_df_from_pickle_safe(spy_data)
        vix_df = _reconstruct_df_from_pickle_safe(vix_data)

        # Import here to avoid issues in worker processes
        from v15.features.tf_extractor import extract_all_tf_features
        from v15.features.validation import validate_features
        from v7.core.channel import detect_channels_multi_window, select_best_channel
        from v15.labels import get_labels_for_position
        from v15.types import ChannelSample

        results = []

        for idx in positions_batch:
            try:
                # Get efficient slice for this position
                start_idx = max(0, idx - SCANNER_LOOKBACK_5MIN)
                tsla_slice = tsla_df.iloc[start_idx:idx]
                spy_slice = spy_df.iloc[start_idx:idx]
                vix_slice = vix_df.iloc[start_idx:idx]
                offset = idx - start_idx

                # Detect channels at this position
                channels = detect_channels_multi_window(tsla_slice, windows=windows)
                best_channel, best_window = select_best_channel(channels)

                if best_channel is None or not best_channel.valid:
                    results.append({'idx': idx, 'skipped': True})
                    continue

                timestamp = tsla_df.index[idx - 1]

                # Extract features
                tf_features = extract_all_tf_features(
                    tsla_df=tsla_slice,
                    spy_df=spy_slice,
                    vix_df=vix_slice,
                    timestamp=timestamp,
                    source_bar_count=idx,
                    include_bar_metadata=True
                )

                # Validate features
                invalid_features = validate_features(tf_features, raise_on_invalid=False)
                if invalid_features:
                    results.append({
                        'idx': idx,
                        'error': f"Invalid features at idx={idx}: {invalid_features[:5]}"
                    })
                    continue

                # Get bar metadata
                resampled_dfs = {'5min': tsla_slice}
                bar_metadata = {
                    '5min': {
                        'bar_completion_pct': 1.0,
                        'bars_in_partial': 0,
                        'total_bars': len(tsla_slice),
                        'is_partial': False,
                    }
                }

                # Look up labels for all windows and timeframes
                labels_per_window = {}
                label_hits = 0
                label_misses = 0

                for window in windows:
                    labels_per_window[window] = {}
                    for tf in timeframes:
                        labels = get_labels_for_position(labeled_map, tsla_df, idx, tf, window)
                        labels_per_window[window][tf] = labels
                        if labels is not None:
                            label_hits += 1
                        else:
                            label_misses += 1

                # Create sample
                sample = ChannelSample(
                    timestamp=timestamp,
                    channel_end_idx=idx,
                    tf_features=tf_features,
                    labels_per_window=labels_per_window,
                    bar_metadata=bar_metadata,
                    best_window=best_window
                )

                results.append({
                    'idx': idx,
                    'sample': sample,
                    'label_hits': label_hits,
                    'label_misses': label_misses,
                })

            except Exception as e:
                results.append({
                    'idx': idx,
                    'error': f"{type(e).__name__}: {str(e)}",
                    'traceback': traceback.format_exc(),
                })

        return results

    except Exception as e:
        # If batch-level error, return error for all positions
        return [{
            'idx': -1,
            'error': f"Batch error: {type(e).__name__}: {str(e)}",
            'traceback': traceback.format_exc(),
        }]


def _save_partial_results(samples: List, output_path: str, suffix: str = "_partial"):
    """
    Save partial results during graceful shutdown.

    Args:
        samples: List of ChannelSample objects collected so far
        output_path: Original output path (will add suffix)
        suffix: Suffix to add to filename (default: "_partial")
    """
    if not samples or not output_path:
        return

    # Create partial output path
    if '.' in output_path:
        base, ext = output_path.rsplit('.', 1)
        partial_path = f"{base}{suffix}.{ext}"
    else:
        partial_path = f"{output_path}{suffix}"

    try:
        with open(partial_path, 'wb') as f:
            pickle.dump(samples, f)
        print(f"\n[SAVED] Partial results ({len(samples)} samples) saved to: {partial_path}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save partial results: {e}")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ScanConfig:
    """Configuration for scanner workers."""
    step: int = SCANNER_CONFIG.get('step', 10)
    warmup_bars: int = SCANNER_CONFIG.get('warmup_bars', 32760)
    forward_bars: int = SCANNER_CONFIG.get('forward_bars', 8000)
    workers: int = SCANNER_CONFIG.get('workers', 4)
    validate_features: bool = True  # Always validate - no silent failures
    chunk_size: int = 100


# Expected feature count (8,665 total)
EXPECTED_FEATURE_COUNT = TOTAL_FEATURES


# =============================================================================
# Feature Validation - LOUD Failures
# =============================================================================

def validate_sample_features(
    tf_features: Dict[str, float],
    timestamp: pd.Timestamp,
    idx: int
) -> Dict[str, float]:
    """
    Validate extracted features. Raises on any invalid value.

    Args:
        tf_features: Dictionary of feature name -> value
        timestamp: Sample timestamp for error messages
        idx: Sample index for error messages

    Returns:
        Validated features dict

    Raises:
        InvalidFeatureError: If any feature is NaN, Inf, or wrong type
        ValidationError: If feature count is wrong
    """
    # Check feature count
    n_features = len(tf_features)
    if n_features == 0:
        raise FeatureExtractionError(
            f"No features extracted at idx={idx}, timestamp={timestamp}"
        )

    # Validate each feature - LOUD failures
    invalid_features = validate_features(tf_features, raise_on_invalid=True)

    # This line only reached if validation passed
    return tf_features


def validate_bar_metadata(
    metadata: Dict[str, Any],
    tf: str,
    idx: int
) -> None:
    """
    Validate bar metadata from resampling.

    Args:
        metadata: Bar metadata dict from resample_with_partial
        tf: Timeframe string
        idx: Sample index for error messages

    Raises:
        ResamplingError: If metadata is invalid
    """
    required_keys = ['bar_completion_pct', 'bars_in_partial', 'total_bars']

    for key in required_keys:
        if key not in metadata:
            raise ResamplingError(
                f"Missing required metadata key '{key}' for {tf} at idx={idx}"
            )

    # Check completion percentage is valid
    completion = metadata['bar_completion_pct']
    if not isinstance(completion, (int, float)) or completion < 0 or completion > 1:
        raise ResamplingError(
            f"Invalid bar_completion_pct={completion} for {tf} at idx={idx}"
        )


# =============================================================================
# Resampling with Partial Bar Support
# =============================================================================

def resample_all_timeframes(
    df: pd.DataFrame,
    idx: int,
    include_metadata: bool = True
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Any]]]:
    """
    Resample DataFrame to all timeframes, keeping partial bars.

    Args:
        df: Base 5min OHLCV DataFrame
        idx: Sample index for error messages
        include_metadata: Whether to include bar metadata

    Returns:
        Tuple of (resampled_dfs, bar_metadata_by_tf)

    Raises:
        ResamplingError: If resampling fails
    """
    resampled_dfs: Dict[str, pd.DataFrame] = {'5min': df}
    bar_metadata: Dict[str, Dict[str, Any]] = {}

    # 5min is the base - calculate its "metadata"
    bar_metadata['5min'] = {
        'bar_completion_pct': 1.0,
        'bars_in_partial': 0,
        'total_bars': len(df),
        'is_partial': False,
    }

    for tf in TIMEFRAMES[1:]:  # Skip 5min
        # Map TF names to pandas resample rules (10 TFs, no 3month)
        tf_map = {
            '15min': '15min',
            '30min': '30min',
            '1h': '1h',
            '2h': '2h',
            '3h': '3h',
            '4h': '4h',
            'daily': '1D',
            'weekly': '1W',
            'monthly': '1MS',
        }

        resample_rule = tf_map.get(tf, tf)

        # Use resample_with_partial to keep incomplete bars
        resampled, metadata = resample_with_partial(df, resample_rule)

        if include_metadata:
            validate_bar_metadata(metadata, tf, idx)

        resampled_dfs[tf] = resampled
        bar_metadata[tf] = metadata

    return resampled_dfs, bar_metadata


def extract_bar_metadata_features(
    bar_metadata: Dict[str, Dict[str, Any]]
) -> Dict[str, float]:
    """
    Convert bar metadata into features for each TF.

    Creates 3 features per TF:
    - {tf}_bar_completion_pct
    - {tf}_bars_in_partial
    - {tf}_complete_bars

    Total: 30 features (10 TFs * 3 features)

    Args:
        bar_metadata: Dict mapping TF -> metadata dict

    Returns:
        Dict of feature name -> value
    """
    features: Dict[str, float] = {}

    for tf in TIMEFRAMES:
        meta = bar_metadata.get(tf, {})

        # Extract metadata values with defaults
        completion = meta.get('bar_completion_pct', 1.0)
        bars_in_partial = meta.get('bars_in_partial', 0)
        total_bars = meta.get('total_bars', 0)
        is_partial = meta.get('is_partial', False)

        # Complete bars = total - 1 if partial, else total
        complete_bars = total_bars - 1 if is_partial else total_bars

        features[f'{tf}_bar_completion_pct'] = float(completion)
        features[f'{tf}_bars_in_partial'] = float(bars_in_partial)
        features[f'{tf}_complete_bars'] = float(complete_bars)

    return features


# =============================================================================
# Efficient Slicing Helper
# =============================================================================

def _get_efficient_slice(
    df: pd.DataFrame,
    idx: int,
    include_forward: bool = False,
    forward_bars: int = 0
) -> Tuple[pd.DataFrame, int]:
    """
    Get efficient data slice for feature extraction.

    Instead of passing ALL historical data from the beginning to idx,
    this function returns only the slice needed for feature extraction.
    This reduces memory usage and resampling costs by ~60x for intraday TFs.

    Args:
        df: Full DataFrame with DatetimeIndex
        idx: Current position index (exclusive end for lookback)
        include_forward: Whether to include forward data for labels
        forward_bars: Number of forward bars to include (only used if include_forward=True)

    Returns:
        Tuple of (sliced_df, offset) where:
        - sliced_df: The efficiently sliced DataFrame
        - offset: The new idx position within sliced_df (use offset instead of idx)

    Example:
        If idx=50000 and SCANNER_LOOKBACK_5MIN=32760:
        - start_idx = 50000 - 32760 = 17240
        - sliced_df = df.iloc[17240:50000]  (32760 bars instead of 50000)
        - offset = 32760 (position of idx within slice)
    """
    # Use SCANNER_LOOKBACK_5MIN as the lookback (covers all TFs)
    # This ensures we have enough data for monthly TF feature extraction
    start_idx = max(0, idx - SCANNER_LOOKBACK_5MIN)

    if include_forward:
        end_idx = min(len(df), idx + forward_bars)
    else:
        end_idx = idx

    sliced = df.iloc[start_idx:end_idx]
    offset = idx - start_idx  # New idx position in sliced data

    return sliced, offset


def _get_optimal_workers(requested: Optional[int] = None) -> int:
    """
    Get optimal number of worker processes.

    Args:
        requested: User-requested worker count, or None for auto-detect

    Returns:
        Number of workers to use
    """
    if requested is not None:
        return max(1, requested)

    # Auto-detect: use CPU count but leave 1 core free for main process
    try:
        cpus = cpu_count()
        # Use at most cpus-1, minimum 1, maximum 8 (diminishing returns beyond)
        return max(1, min(cpus - 1, 8))
    except:
        return 4  # Safe default


# =============================================================================
# Main Scanner Function
# =============================================================================

def scan_channels_two_pass(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    step: int = 10,
    warmup_bars: int = 32760,
    channel_detection_step: int = 1,
    max_samples: Optional[int] = None,
    workers: int = 4,
    batch_size: int = 8,
    progress: bool = True,
    strict: bool = True,
    output_path: Optional[str] = None
) -> List[ChannelSample]:
    """
    Scan for channels using the efficient two-pass labeling system.

    TWO-PASS APPROACH:
    1. **Pre-compute (Pass 1)**: Run detect_all_channels() once on entire dataset
    2. **Label (Pass 2)**: Run generate_all_labels() to label all channels at once
    3. **Scan**: For each position, detect channels and look up labels from map (O(1))

    This approach pre-computes all channels once, then performs O(1) lookups for labels
    instead of scanning forward thousands of bars per sample.

    Args:
        tsla_df: TSLA OHLCV DataFrame with DatetimeIndex
        spy_df: SPY OHLCV DataFrame aligned to TSLA
        vix_df: VIX OHLCV DataFrame aligned to TSLA
        step: Number of bars between samples (default 10)
        warmup_bars: Minimum bars required before first sample (default 32760)
        channel_detection_step: Step for channel detection in Pass 1 (default 1)
        max_samples: Maximum number of samples to generate (default None = unlimited)
        workers: Number of parallel workers (default 4)
        batch_size: Positions per batch for parallel processing (default 8)
        progress: Show progress bar (default True)
        strict: If True, raise on any errors. If False, skip failed samples (default True)
        output_path: Optional output path for saving partial results on interrupt

    Returns:
        List of ChannelSample objects with tf_features and labels

    Raises:
        DataLoadError: If data validation fails
        ValidationError: If strict=True and any sample fails validation
    """
    global _shutdown_requested
    _shutdown_requested = False

    # Set up signal handler for graceful shutdown
    original_sigint = signal.signal(signal.SIGINT, _signal_handler)
    original_sigterm = signal.signal(signal.SIGTERM, _signal_handler)

    n_bars = len(tsla_df)

    # Check memory availability
    _check_memory_availability(workers, n_bars)

    # Calculate position range
    start_idx = warmup_bars
    end_idx = n_bars  # No forward buffer needed - labels come from map

    if start_idx >= end_idx:
        raise DataLoadError(
            f"Not enough data. Have {n_bars} bars, need at least "
            f"{warmup_bars} for warmup."
        )

    # Calculate estimated speedup (conservative estimate based on workers)
    # Amdahl's law with ~80% parallelizable workload
    parallel_enabled = workers > 1
    if parallel_enabled:
        parallel_fraction = 0.80
        estimated_speedup = 1.0 / ((1.0 - parallel_fraction) + (parallel_fraction / workers))
    else:
        estimated_speedup = 1.0

    print("=" * 60)
    if parallel_enabled:
        print("V15 Channel Scanner - Parallel Processing Enabled")
    else:
        print("V15 Channel Scanner - Sequential Mode")
    print("=" * 60)
    print(f"  Workers: {workers} {'(parallel)' if parallel_enabled else '(sequential)'}")
    print(f"  Batch size: {batch_size} positions")
    print(f"  Estimated speedup: ~{estimated_speedup:.1f}x")

    # =========================================================================
    # PASS 1: Pre-compute all channels across the entire dataset
    # =========================================================================
    print("\n[PASS 1] Detecting all channels across dataset...")
    print(f"  Timeframes: {list(TIMEFRAMES)}")
    print(f"  Windows: {list(STANDARD_WINDOWS)}")
    print(f"  Channel detection step: {channel_detection_step}")

    pass1_start = time.time()
    channel_map: ChannelMap = detect_all_channels(
        df=tsla_df,
        timeframes=list(TIMEFRAMES),
        windows=list(STANDARD_WINDOWS),
        step=channel_detection_step,
        min_cycles=1,
        min_gap_bars=5,
        progress_callback=None
    )
    pass1_time = time.time() - pass1_start

    # Count channels detected
    total_channels = sum(len(chs) for chs in channel_map.values())
    print(f"  Detected {total_channels} channels across all TF/window combinations")
    print(f"  Pass 1 time: {pass1_time:.1f}s")
    print("\n[PASS 1] Complete!")
    print("\n[PASS 2] Generating labels from channel map...")

    # =========================================================================
    # PASS 2: Generate labels for all channels
    # =========================================================================
    # Progress callback for Pass 2
    # Accepts: tf (str), window (int), progress_pct (float 0-1)
    def pass2_progress_callback(tf: str, window: int, progress_pct: float):
        pct = progress_pct * 100.0
        if pct >= 100.0 or int(pct) % 10 == 0:
            print(f"  [PASS 2] {tf} window={window}: {pct:.1f}% complete")

    pass2_start = time.time()
    labeled_map: LabeledChannelMap = generate_all_labels(
        channel_map=channel_map,
        progress_callback=pass2_progress_callback
    )
    pass2_time = time.time() - pass2_start

    # Count labeled channels
    total_labeled = sum(len(lcs) for lcs in labeled_map.values())
    valid_labels = sum(
        1 for lcs in labeled_map.values()
        for lc in lcs if lc.labels.direction_valid
    )
    print(f"  Labeled {total_labeled} channels")
    print(f"  Valid direction labels: {valid_labels}")
    print(f"  Pass 2 time: {pass2_time:.1f}s")

    # =========================================================================
    # SCAN: Process each position using lazy label lookups
    # =========================================================================
    positions = list(range(start_idx, end_idx, step))

    # Limit positions if max_samples is specified
    if max_samples is not None and len(positions) > max_samples:
        positions = positions[:max_samples]
        print(f"\n[SCAN] Limited to {max_samples} positions (max_samples specified)")

    total_positions = len(positions)

    print("\n[SCANNING] Starting sample generation (lazy label lookups)...")
    print(f"  Positions to scan: {len(positions)}")
    print(f"  Step size: {step}")
    print(f"  Position range: {start_idx} to {end_idx}")
    print(f"  Expected features per sample: {EXPECTED_FEATURE_COUNT}")
    print(f"  Processing mode: {'PARALLEL' if workers > 1 else 'SEQUENTIAL'}")
    if workers > 1:
        print(f"  Workers: {workers}, Batch size: {batch_size}")

    samples = []
    errors = []
    valid_count = 0
    skipped_count = 0
    label_hit_count = 0
    label_miss_count = 0

    scan_start = time.time()

    try:
        # Convert DataFrames to pickle-safe format ONCE before processing
        tsla_data = _convert_df_to_pickle_safe(tsla_df)
        spy_data = _convert_df_to_pickle_safe(spy_df)
        vix_data = _convert_df_to_pickle_safe(vix_df)

        # Create batches of positions
        batch_size_actual = batch_size or 8
        batches = [positions[i:i+batch_size_actual] for i in range(0, len(positions), batch_size_actual)]

        # Prepare args for each batch
        batch_args = [
            (batch, tsla_data, spy_data, vix_data, labeled_map,
             list(TIMEFRAMES), list(STANDARD_WINDOWS))
            for batch in batches
        ]

        total_batches = len(batches)
        print(f"  Total batches: {total_batches}")

        if workers > 1:
            # =========================================================================
            # PARALLEL PROCESSING with Pool
            # =========================================================================
            print(f"\n  Starting parallel processing with {workers} workers...")

            with Pool(processes=workers) as pool:
                # Use imap_unordered for better performance (order doesn't matter, we sort later)
                results_iter = pool.imap_unordered(_process_position_batch, batch_args)

                # Process results as they come in
                batch_iterator = tqdm(results_iter, total=total_batches,
                                     desc="Processing batches", unit="batch") if progress else results_iter

                batches_processed = 0
                for batch_results in batch_iterator:
                    # Check for shutdown request
                    if _shutdown_requested:
                        print(f"\n[INTERRUPT] Stopping at batch {batches_processed}/{total_batches}")
                        pool.terminate()
                        break

                    # Aggregate results from this batch
                    for result in batch_results:
                        if result.get('sample'):
                            samples.append(result['sample'])
                            valid_count += 1
                            label_hit_count += result.get('label_hits', 0)
                            label_miss_count += result.get('label_misses', 0)
                        elif result.get('error'):
                            errors.append(result['error'])
                            if strict:
                                pool.terminate()
                                raise ValidationError(result['error'])
                        elif result.get('skipped'):
                            skipped_count += 1

                    batches_processed += 1

        else:
            # =========================================================================
            # SEQUENTIAL PROCESSING (fallback when workers=1)
            # =========================================================================
            print(f"\n  Running in sequential mode...")

            batch_iterator = tqdm(batch_args, desc="Processing batches",
                                 unit="batch") if progress else batch_args

            for batch_arg in batch_iterator:
                # Check for shutdown request
                if _shutdown_requested:
                    print(f"\n[INTERRUPT] Stopping scan")
                    break

                # Process batch sequentially
                batch_results = _process_position_batch(batch_arg)

                # Aggregate results from this batch
                for result in batch_results:
                    if result.get('sample'):
                        samples.append(result['sample'])
                        valid_count += 1
                        label_hit_count += result.get('label_hits', 0)
                        label_miss_count += result.get('label_misses', 0)
                    elif result.get('error'):
                        errors.append(result['error'])
                        if strict:
                            raise ValidationError(result['error'])
                    elif result.get('skipped'):
                        skipped_count += 1

    except KeyboardInterrupt:
        print("\n[INTERRUPT] KeyboardInterrupt received. Saving partial results...")
        _shutdown_requested = True

    finally:
        # Restore original signal handlers
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)

        # Save partial results if interrupted
        if _shutdown_requested and output_path:
            _save_partial_results(samples, output_path)

    scan_time = time.time() - scan_start
    total_time = pass1_time + pass2_time + scan_time

    # Sort samples by index
    samples.sort(key=lambda s: s.channel_end_idx)

    # Print summary
    print(f"\n{'=' * 60}")
    if _shutdown_requested:
        print("SCAN INTERRUPTED!")
    else:
        print("SCAN COMPLETE!")
    print(f"{'=' * 60}")
    print(f"  Total positions scanned: {total_positions}")
    print(f"  Valid samples found: {valid_count}")
    print(f"  Skipped (no channel): {skipped_count}")
    print(f"  Errors: {len(errors)}")

    if valid_count > 0:
        avg_features = sum(len(s.tf_features) for s in samples) / valid_count
        print(f"  Average features per sample: {avg_features:.0f}")

    if total_positions > 0:
        skip_rate = 100 * (total_positions - valid_count) / total_positions
        print(f"  Skip rate: {skip_rate:.1f}%")

    # Label lookup stats
    total_lookups = label_hit_count + label_miss_count
    if total_lookups > 0:
        hit_rate = 100 * label_hit_count / total_lookups
        print(f"\n  Label lookup stats:")
        print(f"    Total lookups: {total_lookups}")
        print(f"    Hits: {label_hit_count} ({hit_rate:.1f}%)")
        print(f"    Misses: {label_miss_count} ({100 - hit_rate:.1f}%)")

    # Timing summary
    print(f"\n  Timing breakdown:")
    print(f"    Pass 1 (channel detection): {pass1_time:.1f}s")
    print(f"    Pass 2 (label generation):  {pass2_time:.1f}s")
    print(f"    Scan loop:                  {scan_time:.1f}s")
    print(f"    Total:                      {total_time:.1f}s")

    print(f"\n[COMPLETE] Scanned {valid_count} samples in {total_time:.1f} seconds")

    if errors and not strict:
        print(f"\nWarning: {len(errors)} errors occurred (strict=False)")
        for i, err in enumerate(errors[:5]):
            print(f"  [{i+1}] {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    # Note about window selection strategies for training
    print(f"\n{'=' * 60}")
    print("WINDOW SELECTION STRATEGIES")
    print(f"{'=' * 60}")
    print("When training, you can specify a window selection strategy:")
    print("  --strategy bounce_first    : Select window with most complete cycles (default)")
    print("  --strategy label_validity  : Prioritize windows with valid labels")
    print("  --strategy balanced_score  : Balance between cycle count and label quality")
    print("  --strategy quality_score   : Weight by overall channel quality metrics")
    print("  --strategy learned         : Use learned window selection (requires --end-to-end)")
    print("\nFor end-to-end learning:")
    print("  --end-to-end               : Enable differentiable window selection")
    print("  --window-selection-weight  : Loss weight for window selection (default: 0.1)")

    return samples


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the scanner."""
    # Setup multiprocessing for cross-platform compatibility
    _setup_multiprocessing()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='V15 Two-Pass Channel Scanner')
    parser.add_argument('--step', type=int, default=100,
                        help='Step size for sample generation (default: 100)')
    parser.add_argument('--channel-step', type=int, default=5,
                        help='Step size for channel detection in Pass 1 (default: 5)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to generate (default: unlimited)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path for samples (pickle format)')
    parser.add_argument('--workers', type=int, default=None,
        help='Number of worker processes (default: auto-detect CPU count)')
    parser.add_argument('--batch-size', type=int, default=8,
        help='Positions per batch for parallel processing (default: 8)')
    parser.add_argument('--no-parallel', action='store_true',
        help='Disable parallel processing (run sequentially)')
    args = parser.parse_args()

    # Determine parallelization settings
    if args.no_parallel:
        workers = 1
        batch_size = 1
        parallel_enabled = False
    else:
        workers = _get_optimal_workers(args.workers)
        batch_size = args.batch_size
        parallel_enabled = True

    # Calculate estimated speedup (conservative estimate based on workers)
    # Amdahl's law with ~80% parallelizable workload
    if workers > 1:
        parallel_fraction = 0.80
        estimated_speedup = 1.0 / ((1.0 - parallel_fraction) + (parallel_fraction / workers))
    else:
        estimated_speedup = 1.0

    # Print parallel processing header
    print("=" * 60)
    if parallel_enabled:
        print("V15 Channel Scanner - Parallel Processing Enabled")
    else:
        print("V15 Channel Scanner - Sequential Mode")
    print("=" * 60)

    if parallel_enabled:
        worker_source = "(auto-detected)" if args.workers is None else "(user-specified)"
        print(f"  Workers: {workers} {worker_source}")
        print(f"  Batch size: {batch_size} positions")
        print(f"  Estimated speedup: ~{estimated_speedup:.1f}x")
    else:
        print("  Parallel processing disabled (--no-parallel)")

    print(f"\nConfiguration:")
    print(f"  Step size: {args.step}")
    print(f"  Channel detection step: {args.channel_step}")
    print(f"  Max samples: {args.max_samples if args.max_samples else 'unlimited'}")
    print(f"  Output file: {args.output if args.output else 'none (not saving)'}")

    print("\nLoading market data using v15.data.loader...")
    tsla, spy, vix = load_market_data("data")
    print(f"Loaded {len(tsla)} bars")
    print(f"Date range: {tsla.index[0]} to {tsla.index[-1]}")

    # Check memory availability
    _check_memory_availability(workers, len(tsla))

    print(f"\nExpected feature count: {EXPECTED_FEATURE_COUNT}")
    print(f"  - TF-specific: {TOTAL_FEATURES - FEATURE_COUNTS['events_total']}")
    print(f"  - Events: {FEATURE_COUNTS['events_total']}")
    print(f"  - Bar metadata: {FEATURE_COUNTS['bar_metadata_per_tf'] * len(TIMEFRAMES)}")

    print(f"\nRunning TWO-PASS scan (step={args.step})...")
    samples = scan_channels_two_pass(
        tsla, spy, vix,
        step=args.step,
        channel_detection_step=args.channel_step,
        max_samples=args.max_samples,
        workers=workers,
        batch_size=batch_size,
        progress=True,
        strict=True,  # LOUD failures
        output_path=args.output
    )

    print(f"\nGenerated {len(samples)} samples")

    if samples:
        sample = samples[0]
        print(f"\nFirst sample details:")
        print(f"  Timestamp: {sample.timestamp}")
        print(f"  Best window: {sample.best_window}")
        print(f"  Feature count: {len(sample.tf_features)}")

        # Show some feature names
        feature_names = list(sample.tf_features.keys())
        print(f"\n  Sample feature names (first 10):")
        for name in feature_names[:10]:
            print(f"    - {name}: {sample.tf_features[name]:.4f}")

        # Check for bar metadata features
        bar_meta_features = [k for k in feature_names if 'bar_completion' in k]
        if bar_meta_features:
            print(f"\n  Bar metadata features:")
            for name in bar_meta_features[:5]:
                print(f"    - {name}: {sample.tf_features[name]:.4f}")

        print(f"\nLast sample: {samples[-1].timestamp}")

    # Save samples to output file if specified
    if args.output and samples:
        print(f"\nSaving {len(samples)} samples to {args.output}...")
        with open(args.output, 'wb') as f:
            pickle.dump(samples, f)
        print(f"Saved successfully!")


if __name__ == "__main__":
    # This is REQUIRED for multiprocessing on Windows/macOS
    from multiprocessing import freeze_support
    freeze_support()
    main()
