"""
Channel scanning functions for dataset preparation.

This module contains the core scanning logic for finding valid channels
in historical data. It's separated from dataset.py to avoid torch imports
in multiprocessing workers.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import multiprocessing
import time
import threading

from .types import ChannelSample


# =============================================================================
# Worker Process Globals for Parallel Scanning
# =============================================================================

# Worker process globals for parallel scanning
# Store pre-constructed DataFrames instead of numpy arrays to avoid reconstruction overhead
_WORKER_TSLA_DF = None
_WORKER_SPY_DF = None
_WORKER_VIX_DF = None
_WORKER_PROGRESS_COUNTER = None  # Shared counter for progress updates

# Pre-computed resampled DataFrames to avoid redundant resampling across positions
# Key: timeframe string, Value: resampled DataFrame
_WORKER_PRECOMPUTED_TSLA = None  # Dict[str, pd.DataFrame]
_WORKER_PRECOMPUTED_SPY = None   # Dict[str, pd.DataFrame]

# Progress update stride - flush to shared counter every N positions
PROGRESS_STRIDE = 10


def _init_scan_worker(tsla_df, spy_df, vix_df, progress_counter=None,
                       precomputed_tsla=None, precomputed_spy=None):
    """Initialize worker process with pre-constructed DataFrames, progress counter, and pre-computed resampled data."""
    global _WORKER_TSLA_DF, _WORKER_SPY_DF, _WORKER_VIX_DF, _WORKER_PROGRESS_COUNTER
    global _WORKER_PRECOMPUTED_TSLA, _WORKER_PRECOMPUTED_SPY

    try:
        _WORKER_TSLA_DF = tsla_df
        _WORKER_SPY_DF = spy_df
        _WORKER_VIX_DF = vix_df
        _WORKER_PROGRESS_COUNTER = progress_counter
        _WORKER_PRECOMPUTED_TSLA = precomputed_tsla
        _WORKER_PRECOMPUTED_SPY = precomputed_spy

        # Register pre-computed data with the labels module for cache optimization
        if precomputed_tsla is not None or precomputed_spy is not None:
            from .labels import set_precomputed_resampled_data
            set_precomputed_resampled_data(precomputed_tsla, precomputed_spy)
    except Exception as e:
        # Log initialization failure - this will cause the worker to fail on first task
        # but provides visibility into what went wrong
        import sys
        print(f"ERROR: Worker initialization failed: {e}", file=sys.stderr)
        # Re-raise to ensure the executor knows this worker failed to initialize
        raise


def _process_single_position(
    i: int,
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    window: int,
    min_cycles: int,
    max_scan: int,
    return_threshold: int,
    include_history: bool,
    lookforward_bars: int,
    max_forward_5min_bars: int,
    custom_return_thresholds: Optional[Dict[str, int]]
) -> Optional[Tuple[int, ChannelSample]]:
    """
    Process a single position for channel detection.

    This is a standalone function designed to be called by parallel workers.
    It uses pre-constructed DataFrames from worker globals and efficient .iloc slicing.

    Args:
        i: Position index in the data
        tsla_df: Pre-constructed TSLA DataFrame (from worker globals)
        spy_df: Pre-constructed SPY DataFrame (from worker globals)
        vix_df: Pre-constructed VIX DataFrame (from worker globals)
        window: Channel detection window
        min_cycles: Minimum cycles for valid channel
        max_scan: Maximum bars to scan forward for labels
        return_threshold: Bars outside channel to confirm break
        include_history: Whether to include channel history features
        lookforward_bars: Bars to look forward for exit tracking
        max_forward_5min_bars: Forward bars to include for label generation
        custom_return_thresholds: Optional custom return thresholds per TF

    Returns:
        Tuple of (index, ChannelSample) if valid, None otherwise
    """
    # Import here to avoid issues with multiprocessing
    from ..core.channel import detect_channels_multi_window, select_best_channel, STANDARD_WINDOWS
    from ..features.full_features import extract_full_features, extract_all_window_features
    from .labels import generate_labels_multi_window, select_best_window_by_labels

    # Use efficient .iloc slicing to get data windows (no DataFrame reconstruction)
    # Only use data up to position i for channel detection
    tsla_window_df = tsla_df.iloc[:i]
    spy_window_df = spy_df.iloc[:i]
    vix_window_df = vix_df.iloc[:i]

    if len(tsla_window_df) < window or len(spy_window_df) < window:
        return None

    # Detect channels at multiple window sizes on TSLA
    channels = detect_channels_multi_window(tsla_window_df, windows=STANDARD_WINDOWS, min_cycles=min_cycles)
    if not channels:
        return None

    best_channel, best_window = select_best_channel(channels)
    if not best_channel or not best_channel.valid:
        return None

    # Extract features for all valid windows using OPTIMIZED batch extraction
    # This computes shared features once (resampling, VIX, events, window_scores)
    # and reuses them for all windows, saving ~50-60% extraction time
    try:
        valid_windows = [w for w in STANDARD_WINDOWS if w in channels]
        features_per_window = extract_all_window_features(
            tsla_window_df,
            spy_window_df,
            vix_window_df,
            windows=valid_windows,
            include_history=include_history,
            lookforward_bars=lookforward_bars
        )
    except Exception:
        return None

    # If no windows succeeded, return None
    if not features_per_window:
        return None

    # For backward compatibility, keep the 'features' field as best_window features
    features = features_per_window.get(best_window)
    if features is None:
        # If best_window failed, use the smallest available window for determinism
        # (dict iteration order is insertion order in Python 3.7+, but we use min()
        # to be explicit and avoid any ambiguity across different code paths)
        fallback_window = min(features_per_window.keys())
        features = features_per_window[fallback_window]

    # Generate native per-TF labels for all window sizes
    # Need forward data for label generation - use efficient .iloc slicing
    forward_end = min(i + max_forward_5min_bars, len(tsla_df))
    tsla_full_df = tsla_df.iloc[:forward_end]

    try:
        labels_per_window = generate_labels_multi_window(
            df=tsla_full_df,
            channels=channels,
            channel_end_idx_5min=i - 1,
            max_scan=max_scan,
            return_threshold=return_threshold,
            min_cycles=min_cycles,
            custom_return_thresholds=custom_return_thresholds
        )
        best_labels_window = select_best_window_by_labels(labels_per_window)
        labels_per_tf = labels_per_window[best_labels_window]
    except Exception:
        return None

    # Skip if no valid labels were generated for any TF
    if not labels_per_tf or all(v is None for v in labels_per_tf.values()):
        return None

    # Create sample with multi-window channels, labels, and features
    sample = ChannelSample(
        timestamp=tsla_df.index[i - 1],
        channel_end_idx=i - 1,
        channel=best_channel,
        features=features,
        labels=labels_per_tf,
        channels=channels,
        best_window=best_window,
        labels_per_window=labels_per_window,
        per_window_features=features_per_window  # NEW: Multi-window features
    )

    return (i, sample)


def _process_position_batch(
    indices: List[int],
    window: int,
    min_cycles: int,
    max_scan: int,
    return_threshold: int,
    include_history: bool,
    lookforward_bars: int,
    max_forward_5min_bars: int,
    custom_return_thresholds: Optional[Dict[str, int]]
) -> List[Tuple[int, ChannelSample]]:
    """
    Process a batch of positions. Used by parallel workers.

    Reads pre-constructed DataFrames from worker process globals (_WORKER_TSLA_DF, etc.)
    instead of receiving them as parameters, reducing serialization overhead.

    Returns list of (index, sample) tuples for valid samples.
    """
    # Read from worker globals (set by _init_scan_worker)
    global _WORKER_TSLA_DF, _WORKER_SPY_DF, _WORKER_VIX_DF, _WORKER_PROGRESS_COUNTER

    results = []
    local_processed = 0  # Local counter to batch progress updates

    for i in indices:
        result = _process_single_position(
            i=i,
            tsla_df=_WORKER_TSLA_DF,
            spy_df=_WORKER_SPY_DF,
            vix_df=_WORKER_VIX_DF,
            window=window,
            min_cycles=min_cycles,
            max_scan=max_scan,
            return_threshold=return_threshold,
            include_history=include_history,
            lookforward_bars=lookforward_bars,
            max_forward_5min_bars=max_forward_5min_bars,
            custom_return_thresholds=custom_return_thresholds
        )
        if result is not None:
            results.append(result)

        local_processed += 1

        # Flush progress to shared counter every PROGRESS_STRIDE positions
        if _WORKER_PROGRESS_COUNTER is not None and local_processed >= PROGRESS_STRIDE:
            with _WORKER_PROGRESS_COUNTER.get_lock():
                _WORKER_PROGRESS_COUNTER.value += local_processed
            local_processed = 0

    # Flush any remaining progress
    if _WORKER_PROGRESS_COUNTER is not None and local_processed > 0:
        with _WORKER_PROGRESS_COUNTER.get_lock():
            _WORKER_PROGRESS_COUNTER.value += local_processed

    return results


def _scan_sequential(
    tsla_df: pd.DataFrame,
    spy_aligned: pd.DataFrame,
    vix_aligned: pd.DataFrame,
    indices_list: List[int],
    window: int,
    min_cycles: int,
    max_scan: int,
    return_threshold: int,
    include_history: bool,
    lookforward_bars: int,
    max_forward_5min_bars: int,
    custom_return_thresholds: Optional[Dict[str, int]],
    progress: bool
) -> Tuple[List[ChannelSample], int]:
    """
    Sequential scanning implementation (original behavior).

    Returns:
        Tuple of (samples, valid_count)
    """
    from ..core.channel import detect_channels_multi_window, select_best_channel, STANDARD_WINDOWS
    from ..features.full_features import extract_full_features, extract_all_window_features
    from .labels import generate_labels_multi_window, select_best_window_by_labels

    samples = []
    stats = {
        'total_scanned': 0,
        'invalid_channel': 0,
        'feature_failed': 0,
        'label_failed': 0,
        'no_valid_labels': 0,
        'valid_samples': 0,
    }

    end_idx = indices_list[-1] + 1 if indices_list else 0

    # Pre-slice DataFrames to end_idx to avoid repeated full-copy slicing in loop
    tsla_presliced = tsla_df.iloc[:end_idx]
    spy_presliced = spy_aligned.iloc[:end_idx]
    vix_presliced = vix_aligned.iloc[:end_idx]

    indices = indices_list
    if progress:
        indices = tqdm(indices, desc="Scanning channels (sequential)", mininterval=0.5)

    for i in indices:
        stats['total_scanned'] += 1

        # Get data window (slice from pre-sliced data)
        tsla_window = tsla_presliced.iloc[:i]
        spy_window = spy_presliced.iloc[:i]
        vix_window = vix_presliced.iloc[:i]

        if len(tsla_window) < window or len(spy_window) < window:
            continue

        # Detect channels at multiple window sizes on TSLA
        channels = detect_channels_multi_window(tsla_window, windows=STANDARD_WINDOWS, min_cycles=min_cycles)
        if not channels:
            stats['invalid_channel'] += 1
            continue
        best_channel, best_window = select_best_channel(channels)
        if not best_channel or not best_channel.valid:
            stats['invalid_channel'] += 1
            continue

        # Extract features for all valid windows using OPTIMIZED batch extraction
        # This computes shared features once (resampling, VIX, events, window_scores)
        # and reuses them for all windows, saving ~50-60% extraction time
        try:
            valid_windows = [w for w in STANDARD_WINDOWS if w in channels]
            features_per_window = extract_all_window_features(
                tsla_window,
                spy_window,
                vix_window,
                windows=valid_windows,
                include_history=include_history,
                lookforward_bars=lookforward_bars
            )
        except Exception as e:
            if stats['feature_failed'] == 0 and progress:
                tqdm.write(f"Feature extraction failed (first error): {e}")
            stats['feature_failed'] += 1
            continue

        # If no windows succeeded, skip this position
        if not features_per_window:
            stats['feature_failed'] += 1
            continue

        # For backward compatibility, keep the 'features' field as best_window features
        features = features_per_window.get(best_window)
        if features is None:
            # If best_window failed, use the smallest available window for determinism
            fallback_window = min(features_per_window.keys())
            features = features_per_window[fallback_window]

        # Generate native per-TF labels for all window sizes
        try:
            labels_per_window = generate_labels_multi_window(
                df=tsla_df.iloc[:i + max_forward_5min_bars],  # Include enough forward data for all TFs
                channels=channels,
                channel_end_idx_5min=i - 1,  # Channel ends at the last bar of tsla_window
                max_scan=max_scan,
                return_threshold=return_threshold,
                min_cycles=min_cycles,
                custom_return_thresholds=custom_return_thresholds
            )
            best_labels_window = select_best_window_by_labels(labels_per_window)
            labels_per_tf = labels_per_window[best_labels_window]
        except Exception as e:
            # Skip if label generation fails (only log first occurrence)
            stats['label_failed'] += 1
            if stats['label_failed'] == 1 and progress:
                tqdm.write(f"Label generation failed (first error): {e}")
            continue

        # Skip if no valid labels were generated for any TF
        if not labels_per_tf or all(v is None for v in labels_per_tf.values()):
            stats['no_valid_labels'] += 1
            continue

        # Create sample with multi-window channels, labels, and features
        sample = ChannelSample(
            timestamp=tsla_df.index[i - 1],
            channel_end_idx=i - 1,
            channel=best_channel,  # Best channel for backward compat
            features=features,
            labels=labels_per_tf,  # Best window's labels for backward compat
            channels=channels,  # All channels from multi-window detection
            best_window=best_window,  # Best window size
            labels_per_window=labels_per_window,  # All labels for all windows
            per_window_features=features_per_window  # NEW: Multi-window features
        )

        samples.append(sample)
        stats['valid_samples'] += 1

    # Print detailed stats in sequential mode
    if progress and stats['total_scanned'] > 0:
        if stats['invalid_channel'] > 0:
            print(f"  Invalid channels:        {stats['invalid_channel']} ({100*stats['invalid_channel']/stats['total_scanned']:.1f}%)")
        if stats['feature_failed'] > 0:
            print(f"  Feature extraction failed: {stats['feature_failed']}")
        if stats['label_failed'] > 0:
            print(f"  Label generation failed: {stats['label_failed']}")
        if stats['no_valid_labels'] > 0:
            print(f"  No valid TF labels:      {stats['no_valid_labels']} ({100*stats['no_valid_labels']/stats['total_scanned']:.1f}%)")

    return samples, stats['valid_samples']


def _scan_parallel(
    tsla_df: pd.DataFrame,
    spy_aligned: pd.DataFrame,
    vix_aligned: pd.DataFrame,
    indices_list: List[int],
    window: int,
    min_cycles: int,
    max_scan: int,
    return_threshold: int,
    include_history: bool,
    lookforward_bars: int,
    max_forward_5min_bars: int,
    custom_return_thresholds: Optional[Dict[str, int]],
    max_workers: Optional[int],
    progress: bool,
    heartbeat_timeout_sec: Optional[float],
    heartbeat_interval_sec: float,
    fail_on_timeout: bool,
    max_failed_batches: Optional[int] = None
) -> Tuple[List[ChannelSample], int]:
    """
    Parallel scanning implementation using ProcessPoolExecutor.

    Converts DataFrames to numpy arrays to efficiently pass to worker processes.
    Workers process batches of positions to reduce overhead.

    Args:
        max_failed_batches: Optional maximum number of failed batches before raising an error.
                           If None, continue processing regardless of failures. If exceeded,
                           raises RuntimeError with failure count.

    Returns:
        Tuple of (samples, valid_count)
    """
    # Determine number of workers
    cpu_count = multiprocessing.cpu_count()
    if max_workers is None:
        # Leave one core free, cap at 8 workers to avoid memory issues
        max_workers = min(cpu_count - 1, 8)
    max_workers = max(1, max_workers)  # At least 1 worker

    # Pre-slice DataFrames to include all data needed for label generation
    # (up to max index + forward bars). Workers will use .iloc slicing on these.
    max_idx = max(indices_list) + max_forward_5min_bars
    tsla_presliced = tsla_df.iloc[:max_idx]
    spy_presliced = spy_aligned.iloc[:max_idx]
    vix_presliced = vix_aligned.iloc[:max_idx]

    # Split indices into chunks for workers
    # Each chunk should have enough work to amortize process overhead
    total_positions = len(indices_list)
    chunk_size = max(50, total_positions // (max_workers * 4))  # At least 50 positions per chunk
    chunks = [
        indices_list[i:i + chunk_size]
        for i in range(0, total_positions, chunk_size)
    ]

    if progress:
        print(f"Parallel scanning: {total_positions} positions using {max_workers} workers ({len(chunks)} chunks)")

    # Pre-compute resampled DataFrames for all timeframes ONCE
    # This avoids each worker redundantly resampling the same data
    # Each worker will slice these pre-computed DataFrames to the position timestamp
    from ..core.timeframe import TIMEFRAMES, resample_ohlc

    if progress:
        print("Pre-computing resampled data for all timeframes...")

    # Pre-compute all resampled versions (skip 5min as it's the base)
    # Use the pre-sliced DataFrames which already contain all needed data
    precomputed_tsla = {'5min': tsla_presliced}  # 5min is identity
    precomputed_spy = {'5min': spy_presliced}
    for tf in TIMEFRAMES:
        if tf != '5min':
            precomputed_tsla[tf] = resample_ohlc(tsla_presliced, tf)
            precomputed_spy[tf] = resample_ohlc(spy_presliced, tf)

    if progress:
        print(f"  Pre-computed {len(TIMEFRAMES)} timeframes for TSLA and SPY")

    # Process chunks in parallel
    all_results = []

    # Create shared progress counter using multiprocessing context for cross-platform safety
    # Use 'q' type for 64-bit signed integer to handle large position counts
    ctx = multiprocessing.get_context()
    progress_counter = ctx.Value('q', 0) if progress else None

    # Progress bar with granular updates via shared counter
    pbar = None
    if progress:
        pbar = tqdm(total=total_positions, desc="Scanning channels (parallel)")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_init_scan_worker,
        initargs=(tsla_presliced, spy_presliced, vix_presliced, progress_counter,
                  precomputed_tsla, precomputed_spy)
    ) as executor:
        # Submit all chunks (DataFrames are now in worker globals, not passed per-task)
        futures = {
            executor.submit(
                _process_position_batch,
                chunk,
                window,
                min_cycles,
                max_scan,
                return_threshold,
                include_history,
                lookforward_bars,
                max_forward_5min_bars,
                custom_return_thresholds
            ): i for i, chunk in enumerate(chunks)
        }

        # Set up heartbeat monitoring if timeout is specified
        # Use a lock to synchronize access to last_progress_time between threads
        progress_time_lock = threading.Lock()
        last_progress_time = time.monotonic()
        last_counter_value = 0
        monitor_thread = None
        stop_monitoring = threading.Event()
        # Shared flag for timeout - daemon thread sets it, main thread checks and raises
        timeout_flag = threading.Event()
        timeout_message = [None]  # Use list to allow modification in nested function

        def heartbeat_monitor():
            """Monitor thread that checks for progress timeout."""
            while not stop_monitoring.is_set():
                time.sleep(heartbeat_interval_sec)
                if heartbeat_timeout_sec is not None:
                    with progress_time_lock:
                        elapsed = time.monotonic() - last_progress_time
                    if elapsed > heartbeat_timeout_sec:
                        msg = f"WARNING: No progress for {elapsed:.1f}s (timeout: {heartbeat_timeout_sec}s)"
                        if progress:
                            tqdm.write(msg)
                        else:
                            print(msg)

                        if fail_on_timeout:
                            timeout_message[0] = (
                                f"Worker timeout: No progress for {elapsed:.1f} seconds "
                                f"(threshold: {heartbeat_timeout_sec}s)"
                            )
                            timeout_flag.set()
                            stop_monitoring.set()
                            return  # Exit thread cleanly instead of raising

        # Start monitor thread if timeout monitoring is enabled
        if heartbeat_timeout_sec is not None:
            monitor_thread = threading.Thread(target=heartbeat_monitor, daemon=True)
            monitor_thread.start()

        # Track failed batches for proper error reporting
        failed_batch_count = 0
        failed_batch_errors = []

        # Collect results using wait() with timeout for polling progress
        try:
            pending = set(futures.keys())
            poll_interval = 0.1  # Poll shared counter every 100ms

            while pending:
                # Check if timeout flag was set by monitor thread
                if timeout_flag.is_set():
                    raise TimeoutError(timeout_message[0])

                # Wait for any future to complete, with timeout for progress polling
                done, pending = wait(pending, timeout=poll_interval, return_when=FIRST_COMPLETED)

                # Poll shared counter and update progress bar
                if progress_counter is not None and pbar is not None:
                    with progress_counter.get_lock():
                        current_count = progress_counter.value
                    delta = current_count - last_counter_value
                    if delta > 0:
                        pbar.update(delta)
                        last_counter_value = current_count
                        with progress_time_lock:
                            last_progress_time = time.monotonic()

                # Process completed futures
                for future in done:
                    try:
                        batch_results = future.result()
                        all_results.extend(batch_results)
                    except Exception as e:
                        failed_batch_count += 1
                        failed_batch_errors.append(str(e))
                        if progress:
                            tqdm.write(f"Worker error (batch {failed_batch_count}): {e}")

                        # Check fail-fast threshold
                        if max_failed_batches is not None and failed_batch_count >= max_failed_batches:
                            raise RuntimeError(
                                f"Too many worker failures: {failed_batch_count} batches failed "
                                f"(threshold: {max_failed_batches}). Last error: {e}"
                            )

            # Final progress sync - sync to actual counter value, not assumed total
            # This avoids double-counting or missing updates
            if progress_counter is not None and pbar is not None:
                with progress_counter.get_lock():
                    final_count = progress_counter.value
                # Only update by the actual remaining delta from the counter
                final_delta = final_count - last_counter_value
                if final_delta > 0:
                    pbar.update(final_delta)
                # If counter shows fewer than total (due to skipped positions), complete the bar
                if pbar.n < total_positions:
                    pbar.update(total_positions - pbar.n)

            # Warn user about failed batches if any occurred
            if failed_batch_count > 0:
                warning_msg = (
                    f"WARNING: {failed_batch_count} worker batch(es) failed during processing. "
                    f"Results may be incomplete."
                )
                if progress:
                    tqdm.write(warning_msg)
                else:
                    print(warning_msg)

        finally:
            # Stop monitoring thread with timeout based on heartbeat_interval_sec
            # Use 2x interval to allow current sleep to complete plus margin
            if monitor_thread is not None:
                stop_monitoring.set()
                join_timeout = max(2.0, heartbeat_interval_sec * 2)
                monitor_thread.join(timeout=join_timeout)

    # Close progress bar
    if pbar is not None:
        pbar.close()


    # Sort results by index to maintain deterministic order
    all_results.sort(key=lambda x: x[0])

    # Extract samples (discard indices)
    samples = [sample for _, sample in all_results]

    return samples, len(samples)


def scan_valid_channels(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    window: int = 20,
    step: int = 10,
    min_cycles: int = 1,
    max_scan: int = 500,
    return_threshold: int = 20,
    include_history: bool = False,
    lookforward_bars: int = 200,
    progress: bool = True,
    custom_return_thresholds: Optional[Dict[str, int]] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    heartbeat_timeout_sec: Optional[float] = None,
    heartbeat_interval_sec: float = 5.0,
    fail_on_timeout: bool = False,
    max_failed_batches: Optional[int] = None
) -> Tuple[List[ChannelSample], int]:
    """
    Scan through historical data to find all valid channels and generate samples.

    Supports both sequential and parallel processing modes. Parallel mode uses
    ProcessPoolExecutor for significant speedup on multi-core systems.

    Returns:
        Tuple of (samples, min_warmup_bars)
        - samples: List of ChannelSample objects
        - min_warmup_bars: Minimum warmup bars used (for cache metadata)

    This is the main data preparation function. It:
    1. Slides through time with a rolling window
    2. At each position, detects a channel
    3. If valid (has cycles), extracts features and generates labels
    4. Returns list of training samples

    Args:
        tsla_df: TSLA 5min OHLCV data
        spy_df: SPY 5min OHLCV data
        vix_df: VIX daily data
        window: Window size for channel detection
        step: Step size for sliding window (smaller = more samples, slower)
        min_cycles: Minimum cycles to consider channel valid
        max_scan: Maximum bars to scan forward for labels
        return_threshold: Bars outside channel to confirm break
        include_history: Whether to include channel history features (slower)
        lookforward_bars: Bars to look forward for exit tracking
        progress: Show progress bar
        custom_return_thresholds: Optional custom return thresholds per TF
        parallel: If True (default), use parallel processing with ProcessPoolExecutor.
                  Set to False for sequential processing (useful for debugging).
        max_workers: Maximum number of worker processes for parallel mode.
                     If None, defaults to min(cpu_count - 1, 8) to leave one core free.
        heartbeat_timeout_sec: Optional timeout in seconds. If specified and no progress is
                               made within this duration, either warn or fail based on fail_on_timeout.
                               If None, no timeout monitoring is performed.
        heartbeat_interval_sec: Interval in seconds for checking heartbeat timeout (default: 5.0).
        fail_on_timeout: If True, raise TimeoutError when heartbeat timeout is exceeded.
                        If False (default), only print a warning and continue.
        max_failed_batches: Optional maximum number of failed worker batches before raising an error.
                           If None (default), continue processing and warn at the end.
                           If exceeded, raises RuntimeError with failure count.

    Returns:
        List of ChannelSample objects
    """
    # Warmup period: ensure adequate data for all timeframes
    # Need ~32,760 bars (420 trading days) for monthly channels to have window=20 native monthly bars
    # This ensures monthly timeframes have proper statistical validity (20 bars for regression)
    # 3-month will still be weak (~6.7 bars) but acceptable with quality scoring
    min_warmup_bars = max(window, 32760)  # At least 32,760 bars (20 months) or window, whichever larger

    # Calculate forward 5min bars needed for longest timeframe label generation
    # Daily needs 50 daily bars of forward data for label scanning
    # Data includes extended hours (~150-190 5min bars/day, not just 78 regular hours)
    # Use 8000 5min bars to safely get 50+ daily bars with margin for holidays
    # This ensures all TFs (5min through daily) have enough forward data
    max_forward_5min_bars = 8000  # ~50-55 daily bars with extended hours data

    # Align SPY and VIX with TSLA timestamps
    # For SPY (5min), reindex to match TSLA timestamps and forward-fill gaps
    spy_aligned = spy_df.reindex(tsla_df.index, method='ffill')

    # For VIX (daily), forward-fill to match 5min timestamps
    vix_aligned = vix_df.reindex(tsla_df.index, method='ffill')

    # Scan through data with sliding window
    start_idx = min_warmup_bars
    end_idx = len(tsla_df) - max_forward_5min_bars  # Reserve enough forward data for all TFs

    indices_list = list(range(start_idx, end_idx, step))
    total_positions = len(indices_list)

    if total_positions == 0:
        if progress:
            print("No positions to scan (data range too small)")
        return [], min_warmup_bars

    # Determine processing mode
    if parallel and total_positions > 100:  # Only parallelize if enough work
        samples, valid_count = _scan_parallel(
            tsla_df=tsla_df,
            spy_aligned=spy_aligned,
            vix_aligned=vix_aligned,
            indices_list=indices_list,
            window=window,
            min_cycles=min_cycles,
            max_scan=max_scan,
            return_threshold=return_threshold,
            include_history=include_history,
            lookforward_bars=lookforward_bars,
            max_forward_5min_bars=max_forward_5min_bars,
            custom_return_thresholds=custom_return_thresholds,
            max_workers=max_workers,
            progress=progress,
            heartbeat_timeout_sec=heartbeat_timeout_sec,
            heartbeat_interval_sec=heartbeat_interval_sec,
            fail_on_timeout=fail_on_timeout,
            max_failed_batches=max_failed_batches
        )
    else:
        samples, valid_count = _scan_sequential(
            tsla_df=tsla_df,
            spy_aligned=spy_aligned,
            vix_aligned=vix_aligned,
            indices_list=indices_list,
            window=window,
            min_cycles=min_cycles,
            max_scan=max_scan,
            return_threshold=return_threshold,
            include_history=include_history,
            lookforward_bars=lookforward_bars,
            max_forward_5min_bars=max_forward_5min_bars,
            custom_return_thresholds=custom_return_thresholds,
            progress=progress
        )

    # Print summary
    if progress and total_positions > 0:
        print(f"\nChannel scanning summary:")
        print(f"  Total positions scanned: {total_positions}")
        print(f"  Valid samples created:   {valid_count} ({100*valid_count/total_positions:.1f}%)")

    return samples, min_warmup_bars
