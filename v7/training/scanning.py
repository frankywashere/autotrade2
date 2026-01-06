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
_WORKER_TSLA_VALUES = None
_WORKER_TSLA_INDEX = None
_WORKER_SPY_VALUES = None
_WORKER_VIX_VALUES = None
_WORKER_PROGRESS_COUNTER = None  # Shared counter for progress updates

# Progress update stride - flush to shared counter every N positions
PROGRESS_STRIDE = 10


def _init_scan_worker(tsla_values, tsla_index, spy_values, vix_values, progress_counter=None):
    """Initialize worker process with shared data arrays and progress counter."""
    global _WORKER_TSLA_VALUES, _WORKER_TSLA_INDEX, _WORKER_SPY_VALUES, _WORKER_VIX_VALUES, _WORKER_PROGRESS_COUNTER
    _WORKER_TSLA_VALUES = tsla_values
    _WORKER_TSLA_INDEX = tsla_index
    _WORKER_SPY_VALUES = spy_values
    _WORKER_VIX_VALUES = vix_values
    _WORKER_PROGRESS_COUNTER = progress_counter


def _process_single_position(
    i: int,
    tsla_values: np.ndarray,
    tsla_index: pd.DatetimeIndex,
    spy_values: np.ndarray,
    vix_values: np.ndarray,
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
    It receives numpy arrays instead of DataFrames to avoid pickling overhead.

    Args:
        i: Position index in the data
        tsla_values: TSLA OHLCV data as numpy array (columns: open, high, low, close, volume)
        tsla_index: TSLA datetime index
        spy_values: SPY OHLCV data as numpy array
        vix_values: VIX OHLCV data as numpy array
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
    from ..features.full_features import extract_full_features
    from .labels import generate_labels_multi_window, select_best_window_by_labels

    # Reconstruct DataFrames from numpy arrays for this position
    # Only use data up to position i for channel detection
    tsla_window_df = pd.DataFrame(
        tsla_values[:i],
        index=tsla_index[:i],
        columns=['open', 'high', 'low', 'close', 'volume']
    )
    spy_window_df = pd.DataFrame(
        spy_values[:i],
        index=tsla_index[:i],
        columns=['open', 'high', 'low', 'close', 'volume']
    )
    vix_window_df = pd.DataFrame(
        vix_values[:i],
        index=tsla_index[:i],
        columns=['open', 'high', 'low', 'close']
    )

    if len(tsla_window_df) < window or len(spy_window_df) < window:
        return None

    # Detect channels at multiple window sizes on TSLA
    channels = detect_channels_multi_window(tsla_window_df, windows=STANDARD_WINDOWS, min_cycles=min_cycles)
    if not channels:
        return None

    best_channel, best_window = select_best_channel(channels)
    if not best_channel or not best_channel.valid:
        return None

    # Extract features at this point
    try:
        features = extract_full_features(
            tsla_window_df,
            spy_window_df,
            vix_window_df,
            window=best_window,
            include_history=include_history,
            lookforward_bars=lookforward_bars
        )
    except Exception:
        return None

    # Generate native per-TF labels for all window sizes
    # Need forward data for label generation
    forward_end = min(i + max_forward_5min_bars, len(tsla_values))
    tsla_full_df = pd.DataFrame(
        tsla_values[:forward_end],
        index=tsla_index[:forward_end],
        columns=['open', 'high', 'low', 'close', 'volume']
    )

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

    # Create sample with multi-window channels and labels
    sample = ChannelSample(
        timestamp=tsla_index[i - 1],
        channel_end_idx=i - 1,
        channel=best_channel,
        features=features,
        labels=labels_per_tf,
        channels=channels,
        best_window=best_window,
        labels_per_window=labels_per_window
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

    Reads data arrays from worker process globals (_WORKER_TSLA_VALUES, etc.)
    instead of receiving them as parameters, reducing serialization overhead.

    Returns list of (index, sample) tuples for valid samples.
    """
    # Read from worker globals (set by _init_scan_worker)
    global _WORKER_TSLA_VALUES, _WORKER_TSLA_INDEX, _WORKER_SPY_VALUES, _WORKER_VIX_VALUES, _WORKER_PROGRESS_COUNTER

    results = []
    local_processed = 0  # Local counter to batch progress updates

    for i in indices:
        result = _process_single_position(
            i=i,
            tsla_values=_WORKER_TSLA_VALUES,
            tsla_index=_WORKER_TSLA_INDEX,
            spy_values=_WORKER_SPY_VALUES,
            vix_values=_WORKER_VIX_VALUES,
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
    from ..features.full_features import extract_full_features
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

        # Extract features at this point (use best_window for feature extraction)
        try:
            features = extract_full_features(
                tsla_window,
                spy_window,
                vix_window,
                window=best_window,
                include_history=include_history,
                lookforward_bars=lookforward_bars
            )
        except Exception as e:
            # Skip if feature extraction fails (only log first occurrence)
            stats['feature_failed'] += 1
            if stats['feature_failed'] == 1 and progress:
                tqdm.write(f"Feature extraction failed (first error): {e}")
            continue

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

        # Create sample with multi-window channels and labels
        sample = ChannelSample(
            timestamp=tsla_df.index[i - 1],
            channel_end_idx=i - 1,
            channel=best_channel,  # Best channel for backward compat
            features=features,
            labels=labels_per_tf,  # Best window's labels for backward compat
            channels=channels,  # All channels from multi-window detection
            best_window=best_window,  # Best window size
            labels_per_window=labels_per_window  # All labels for all windows
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
    fail_on_timeout: bool
) -> Tuple[List[ChannelSample], int]:
    """
    Parallel scanning implementation using ProcessPoolExecutor.

    Converts DataFrames to numpy arrays to efficiently pass to worker processes.
    Workers process batches of positions to reduce overhead.

    Returns:
        Tuple of (samples, valid_count)
    """
    # Determine number of workers
    cpu_count = multiprocessing.cpu_count()
    if max_workers is None:
        # Leave one core free, cap at 8 workers to avoid memory issues
        max_workers = min(cpu_count - 1, 8)
    max_workers = max(1, max_workers)  # At least 1 worker

    # Convert DataFrames to numpy arrays for efficient pickling
    # Include all data needed for label generation (up to max index + forward bars)
    # Use .to_numpy(copy=False) to avoid unnecessary copying
    max_idx = max(indices_list) + max_forward_5min_bars
    tsla_values = tsla_df.iloc[:max_idx].to_numpy(copy=False)
    tsla_index = tsla_df.index[:max_idx]
    spy_values = spy_aligned.iloc[:max_idx].to_numpy(copy=False)
    # VIX has fewer columns - handle the column difference
    vix_cols = ['open', 'high', 'low', 'close'] if 'open' in vix_aligned.columns else list(vix_aligned.columns[:4])
    vix_values = vix_aligned[vix_cols].iloc[:max_idx].to_numpy(copy=False)

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
        initargs=(tsla_values, tsla_index, spy_values, vix_values, progress_counter)
    ) as executor:
        # Submit all chunks (arrays are now in worker globals, not passed per-task)
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
        last_progress_time = time.monotonic()
        last_counter_value = 0
        monitor_thread = None
        stop_monitoring = threading.Event()

        def heartbeat_monitor():
            """Monitor thread that checks for progress timeout."""
            nonlocal last_progress_time
            while not stop_monitoring.is_set():
                time.sleep(heartbeat_interval_sec)
                if heartbeat_timeout_sec is not None:
                    elapsed = time.monotonic() - last_progress_time
                    if elapsed > heartbeat_timeout_sec:
                        msg = f"WARNING: No progress for {elapsed:.1f}s (timeout: {heartbeat_timeout_sec}s)"
                        if progress:
                            tqdm.write(msg)
                        else:
                            print(msg)

                        if fail_on_timeout:
                            stop_monitoring.set()
                            raise TimeoutError(
                                f"Worker timeout: No progress for {elapsed:.1f} seconds "
                                f"(threshold: {heartbeat_timeout_sec}s)"
                            )

        # Start monitor thread if timeout monitoring is enabled
        if heartbeat_timeout_sec is not None:
            monitor_thread = threading.Thread(target=heartbeat_monitor, daemon=True)
            monitor_thread.start()

        # Collect results using wait() with timeout for polling progress
        try:
            pending = set(futures.keys())
            poll_interval = 0.1  # Poll shared counter every 100ms

            while pending:
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
                        last_progress_time = time.monotonic()

                # Process completed futures
                for future in done:
                    try:
                        batch_results = future.result()
                        all_results.extend(batch_results)
                    except Exception as e:
                        if progress:
                            tqdm.write(f"Worker error: {e}")

            # Final progress sync - catch any remaining counts
            if progress_counter is not None and pbar is not None:
                with progress_counter.get_lock():
                    final_count = progress_counter.value
                remaining = total_positions - pbar.n
                if remaining > 0:
                    pbar.update(remaining)

        finally:
            # Stop monitoring thread
            if monitor_thread is not None:
                stop_monitoring.set()
                monitor_thread.join(timeout=1.0)

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
    fail_on_timeout: bool = False
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
            fail_on_timeout=fail_on_timeout
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
