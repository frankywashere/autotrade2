"""
Two-Pass Label Generation for Channel Break Prediction (v15)

This module implements an efficient two-pass labeling system for channel analysis:

PASS 1 - detect_all_channels():
    Scans the entire dataset once and detects ALL channels for each TF/window
    combination. Channels are stored with timestamps for cross-TF alignment.
    Complexity: O(N) where N is dataset length.

PASS 2 - generate_all_labels():
    For each detected channel, looks up the "next channel" in the map to
    determine break direction and duration. Labels are pre-computed and stored.
    Complexity: O(C) where C is number of channels.

LOOKUP - get_labels_for_position():
    O(log N) binary search lookup of labels for a specific position in the
    dataset. Used by training pipeline to get labels for sample positions.

This two-pass approach is significantly more efficient than the previous
single-pass method which required O(N * max_scan) complexity per sample.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import IntEnum
from multiprocessing import Pool, cpu_count

# Import from existing v7 modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from v7.core.channel import detect_channel, Channel
from v7.core.timeframe import resample_ohlc

# Import ChannelLabels and CrossCorrelationLabels from v15.types
from .types import ChannelLabels, CrossCorrelationLabels

# Import TF_MAX_SCAN for forward scanning limits
from .config import TF_MAX_SCAN

# Import break scanner for sophisticated break detection
from .core.break_scanner import scan_for_break, BreakResult, InsufficientDataError


# =============================================================================
# Enums and Constants
# =============================================================================

class BreakDirection(IntEnum):
    """Direction of channel break."""
    DOWN = 0
    UP = 1


class NewChannelDirection(IntEnum):
    """Direction of the new channel that forms after break."""
    BEAR = 0
    SIDEWAYS = 1
    BULL = 2


@dataclass
class DetectedChannel:
    """
    A channel detected during Pass 1 scanning.

    Attributes:
        start_idx: Index where channel detection window starts (in TF-resampled data)
        end_idx: Index where channel detection window ends (in TF-resampled data)
        direction: Channel direction (BEAR=0, SIDEWAYS=1, BULL=2)
        channel: The full Channel object with all metrics
        tf: Timeframe this channel was detected at
        window: Window size used for detection
        start_timestamp: Timestamp of start_idx (for cross-TF alignment)
        end_timestamp: Timestamp of end_idx (for cross-TF alignment)
    """
    start_idx: int
    end_idx: int
    direction: int  # 0=BEAR, 1=SIDEWAYS, 2=BULL
    channel: Channel
    tf: str
    window: int
    start_timestamp: pd.Timestamp = None
    end_timestamp: pd.Timestamp = None


@dataclass
class LabeledChannel:
    """
    A channel with labels from Pass 2.

    Attributes:
        detected: The DetectedChannel from Pass 1
        labels: The ChannelLabels derived from next channel lookup
        next_channel_idx: Index of the next channel in the sequence (or -1 if none)
    """
    detected: DetectedChannel
    labels: ChannelLabels
    next_channel_idx: int = -1


# Type aliases for the channel maps
ChannelMap = Dict[Tuple[str, int], List[DetectedChannel]]
LabeledChannelMap = Dict[Tuple[str, int], List[LabeledChannel]]


# =============================================================================
# PASS 1: Channel Detection
# =============================================================================

def _detect_tf_window_worker(args):
    """
    Worker function to detect channels for one (tf, window) combination.

    This function is called by Pool.map() for parallel processing.
    It reconstructs the DataFrame from serialized values since DataFrames
    cannot be pickled directly for multiprocessing.

    Args:
        args: Tuple of (tf, window, step, min_cycles, min_gap_bars,
              df_values, df_index, df_columns)

    Returns:
        Tuple of (tf, window, detected_channels, failed_positions)
    """
    tf, window, step, min_cycles, min_gap_bars, df_values, df_index, df_columns = args

    # Reconstruct DataFrame from serialized components
    df_tf = pd.DataFrame(df_values, index=pd.DatetimeIndex(df_index), columns=df_columns)

    detected_channels = []
    failed_positions = []

    if len(df_tf) < window:
        return (tf, window, detected_channels, failed_positions)

    # Scan through the dataset
    scan_positions = list(range(window - 1, len(df_tf), step))
    last_channel_end = -min_gap_bars  # Allow first channel to start immediately

    for end_idx in scan_positions:
        start_idx = end_idx - window + 1

        # Skip if this channel's START overlaps with previous channel's END
        if start_idx <= last_channel_end + min_gap_bars:
            continue

        df_slice = df_tf.iloc[start_idx:end_idx + 1]

        try:
            channel = detect_channel(df_slice, window=window, min_cycles=min_cycles)

            if channel.valid:
                # Get timestamps for cross-TF alignment
                start_ts = df_tf.index[start_idx]
                end_ts = df_tf.index[end_idx]

                detected = DetectedChannel(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    direction=int(channel.direction),
                    channel=channel,
                    tf=tf,
                    window=window,
                    start_timestamp=start_ts,
                    end_timestamp=end_ts
                )
                detected_channels.append(detected)

                # Update last channel end to prevent overlapping
                last_channel_end = end_idx
        except Exception as e:
            # Channel detection failed - log warning and track failed position
            print(f"[WARNING] Channel detection failed: tf={tf}, window={window}, "
                  f"start_idx={start_idx}, end_idx={end_idx}, error={e}")
            failed_positions.append((start_idx, end_idx, str(e)))

    return (tf, window, detected_channels, failed_positions)


def detect_all_channels(
    df: pd.DataFrame,
    timeframes: List[str] = None,
    windows: List[int] = None,
    step: int = 1,
    min_cycles: int = 1,
    min_gap_bars: int = 5,
    progress_callback=None,
    verbose: bool = True,
    workers: int = None
) -> Tuple[ChannelMap, Dict[str, pd.DataFrame]]:
    """
    PASS 1: Detect all channels across entire dataset for all TF/window combinations.

    Scans through the entire dataset and detects channels at regular intervals.
    Uses parallel processing with multiprocessing.Pool to process each (TF, window)
    combination concurrently for significant speedup on multi-core systems.

    Args:
        df: Base 5min OHLCV DataFrame with DatetimeIndex
        timeframes: List of timeframes to scan (defaults to all TIMEFRAMES)
        windows: List of window sizes (defaults to STANDARD_WINDOWS)
        step: Step size in TF bars between channel detection attempts
        min_cycles: Minimum bounces for valid channel
        min_gap_bars: Minimum bars between channel end and next channel start
        progress_callback: Optional callback(tf, window, pct) for progress updates
        verbose: If True, print detailed progress logging
        workers: Number of parallel workers (defaults to cpu_count()-1, no cap)

    Returns:
        Tuple of:
            - ChannelMap: {(tf, window): [DetectedChannel, ...]} sorted by start_idx
            - Dict[str, pd.DataFrame]: Resampled DataFrames keyed by timeframe
    """
    from v15.config import TIMEFRAMES as ALL_TIMEFRAMES, STANDARD_WINDOWS

    if timeframes is None:
        timeframes = ALL_TIMEFRAMES
    if windows is None:
        windows = STANDARD_WINDOWS

    # Pre-resample all timeframes once (sequential - it's fast)
    resampled_dfs: Dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        if tf == '5min':
            resampled_dfs[tf] = df
        else:
            try:
                resampled_dfs[tf] = resample_ohlc(df, tf)
            except Exception as e:
                print(f"[WARNING] Failed to resample timeframe '{tf}': {e}")
                resampled_dfs[tf] = None

    # Calculate total work estimate for verbose logging
    num_tfs = len(timeframes)
    num_windows = len(windows)
    estimated_positions_per_tf = {}
    total_estimated_positions = 0
    for tf in timeframes:
        df_tf = resampled_dfs.get(tf)
        if df_tf is not None and len(df_tf) > 0:
            # Estimate positions: (len - min_window) / step
            min_window = min(windows) if windows else 10
            est_pos = max(0, (len(df_tf) - min_window + 1) // step)
            estimated_positions_per_tf[tf] = est_pos
            total_estimated_positions += est_pos * num_windows
        else:
            estimated_positions_per_tf[tf] = 0

    # Determine number of workers
    # No cap - use workers parameter to limit if needed
    cpus = cpu_count() or 8  # cpu_count() can return None on some systems
    num_workers = workers if workers is not None else max(1, cpus - 1)

    if verbose:
        print(f"[PASS 1] Starting channel detection (PARALLEL with {num_workers} workers):")
        print(f"         TFs: {num_tfs}, Windows: {num_windows}, Step: {step}")
        print(f"         Estimated total scan positions: {total_estimated_positions:,}")
        print()

    # Create work items for parallel processing
    # Each work item is a (tf, window) combination with serialized DataFrame
    work_items = []
    for tf in timeframes:
        df_tf = resampled_dfs.get(tf)
        if df_tf is None or len(df_tf) == 0:
            continue
        for window in windows:
            if len(df_tf) < window:
                continue
            # Serialize DataFrame for multiprocessing (can't pickle DataFrames directly)
            work_items.append((
                tf,
                window,
                step,
                min_cycles,
                min_gap_bars,
                df_tf.values,
                df_tf.index.values,
                list(df_tf.columns)
            ))

    if verbose:
        print(f"[PASS 1] Created {len(work_items)} work items for parallel processing...")

    # Initialize channel_map with empty lists for all combinations
    # (some may not have work items if df is too short)
    channel_map: ChannelMap = {}
    for tf in timeframes:
        for window in windows:
            channel_map[(tf, window)] = []

    # Process work items in parallel using Pool.map()
    if work_items:
        with Pool(processes=num_workers) as pool:
            results = pool.map(_detect_tf_window_worker, work_items)

        # Collect results into channel_map
        total_channels_found = 0
        total_failed_positions = 0
        for tf, window, detected_channels, failed_positions in results:
            channel_map[(tf, window)] = detected_channels
            num_channels = len(detected_channels)
            total_channels_found += num_channels
            total_failed_positions += len(failed_positions)
            if verbose:
                fail_info = f" ({len(failed_positions)} failed)" if failed_positions else ""
                print(f"  TF={tf}, Window={window}: found {num_channels} channels{fail_info}")

            # Progress callback (approximate since parallel)
            if progress_callback:
                progress_callback(tf, window, 1.0)
    else:
        total_channels_found = 0
        total_failed_positions = 0

    if verbose:
        print()
        print(f"[PASS 1] Complete: Found {total_channels_found} total channels across all TF/window combinations")
        if total_failed_positions > 0:
            print(f"         Total failed positions: {total_failed_positions}")

    return channel_map, resampled_dfs


# =============================================================================
# PASS 2: Label Generation
# =============================================================================

def label_channel_from_map(
    channel_map: ChannelMap,
    tf: str,
    window: int,
    channel_idx: int
) -> Tuple[ChannelLabels, int]:
    """
    Label a single channel by looking up the next channel in the map.

    The break direction is determined by how price ACTUALLY broke out of the
    current channel - comparing the price at the start of the next channel
    against the current channel's projected upper/lower bounds:
    - If price is ABOVE current channel's projected upper bound -> break_direction = UP
    - If price is BELOW current channel's projected lower bound -> break_direction = DOWN
    - If within bounds (rare), fall back to price movement direction
    - If no next channel (end of data) -> direction_valid = False

    Args:
        channel_map: The channel map from detect_all_channels()
        tf: Timeframe
        window: Window size
        channel_idx: Index of channel in the list for this (tf, window)

    Returns:
        Tuple of (ChannelLabels, next_channel_idx)
        next_channel_idx is -1 if no next channel found
    """
    key = (tf, window)
    channels = channel_map.get(key, [])

    if channel_idx < 0 or channel_idx >= len(channels):
        # Invalid index
        return ChannelLabels(
            duration_bars=0,
            break_direction=int(BreakDirection.UP),
            next_channel_direction=int(NewChannelDirection.SIDEWAYS),
            permanent_break=False,
            timeframe=tf,
            duration_valid=False,
            direction_valid=False,
            next_channel_valid=False
        ), -1

    current = channels[channel_idx]

    # Look for next channel
    if channel_idx + 1 < len(channels):
        next_channel = channels[channel_idx + 1]
        next_idx = channel_idx + 1

        # Duration is the gap between channels
        duration_bars = next_channel.start_idx - current.end_idx

        # Get the next channel's direction for next_channel_direction label
        # This tells us what type of channel formed after the break (BEAR/SIDEWAYS/BULL)
        next_dir = next_channel.direction

        # Determine break direction by comparing price against current channel's
        # projected bounds at the point where the next channel starts.
        #
        # The channel's linear regression is: center = slope * x + intercept
        # where x is the bar index relative to the start of the channel window.
        # At the end of the current channel, x = window - 1.
        # The next channel starts 'duration_bars' after the current channel ends.
        # So we project to x = window - 1 + duration_bars.
        #
        # Upper bound = center + 2 * std_dev
        # Lower bound = center - 2 * std_dev

        curr_channel = current.channel
        next_ch_data = next_channel.channel

        # Get the price at the start of the next channel
        # The next channel's close array starts at its start_idx
        if next_ch_data.close is not None and len(next_ch_data.close) > 0:
            next_start_price = next_ch_data.close[0]
        else:
            # Fallback: use direction-based logic if no price data
            next_start_price = None

        # Calculate projected bounds of current channel at next channel's start
        if next_start_price is not None and curr_channel.slope is not None:
            # Project current channel forward to where next channel starts
            # x at end of current channel = window - 1
            # x at start of next channel = (window - 1) + duration_bars
            projection_x = curr_channel.window - 1 + duration_bars

            # Projected center line at that point
            projected_center = curr_channel.slope * projection_x + curr_channel.intercept

            # Use 2 * std_dev for channel width (standard +-2 sigma bounds)
            std_multiplier = 2.0
            projected_upper = projected_center + std_multiplier * curr_channel.std_dev
            projected_lower = projected_center - std_multiplier * curr_channel.std_dev

            # Determine break direction based on where next channel starts
            # relative to the current channel's projected bounds
            if next_start_price > projected_upper:
                # Price broke above the upper bound -> UP break
                break_direction = BreakDirection.UP
            elif next_start_price < projected_lower:
                # Price broke below the lower bound -> DOWN break
                break_direction = BreakDirection.DOWN
            else:
                # Price is within projected bounds (rare) - use price movement
                # Compare next channel's start price vs current channel's end price
                if curr_channel.close is not None and len(curr_channel.close) > 0:
                    curr_end_price = curr_channel.close[-1]
                    if next_start_price > curr_end_price:
                        break_direction = BreakDirection.UP
                    else:
                        break_direction = BreakDirection.DOWN
                else:
                    # Last resort: use next channel direction
                    if next_dir == NewChannelDirection.BULL:
                        break_direction = BreakDirection.UP
                    else:
                        break_direction = BreakDirection.DOWN
        else:
            # Fallback if no price data available: use next channel direction
            if next_dir == NewChannelDirection.BULL:
                break_direction = BreakDirection.UP
            elif next_dir == NewChannelDirection.BEAR:
                break_direction = BreakDirection.DOWN
            else:
                # Sideways - compare intercepts
                if next_ch_data.intercept > curr_channel.intercept:
                    break_direction = BreakDirection.UP
                else:
                    break_direction = BreakDirection.DOWN

        return ChannelLabels(
            duration_bars=duration_bars,
            break_direction=int(break_direction),
            next_channel_direction=next_dir,
            permanent_break=True,
            timeframe=tf,
            duration_valid=True,
            direction_valid=True,
            next_channel_valid=True
        ), next_idx

    else:
        # No next channel - end of data
        return ChannelLabels(
            duration_bars=0,
            break_direction=int(BreakDirection.UP),  # Default
            next_channel_direction=int(NewChannelDirection.SIDEWAYS),
            permanent_break=False,
            timeframe=tf,
            duration_valid=False,
            direction_valid=False,
            next_channel_valid=False
        ), -1


def label_channel_forward_scan(
    detected: DetectedChannel,
    resampled_df: pd.DataFrame,
    max_scan: int,
    next_channel_direction: Optional[int] = None
) -> ChannelLabels:
    """
    Label a channel using forward bar scanning to find the first break.

    Uses the sophisticated scan_for_break() function from break_scanner.py
    which provides:
    - Accurate break detection using HIGH/LOW for exits (not just close)
    - Return-to-channel tracking (false break detection)
    - Multiple exit event tracking for durability analysis
    - Break magnitude in standard deviations

    Args:
        detected: The DetectedChannel from Pass 1
        resampled_df: The resampled DataFrame for this timeframe
        max_scan: Maximum number of bars to scan forward (from TF_MAX_SCAN)
        next_channel_direction: Optional direction of next channel (0=BEAR, 1=SIDEWAYS, 2=BULL).
                                If provided (e.g., from hybrid method), populates next_channel_direction.

    Returns:
        ChannelLabels with all break scan fields populated:
        - bars_to_first_break: When the first break occurred
        - break_direction: Direction of break (0=DOWN, 1=UP)
        - break_magnitude: How far outside bounds (in std devs)
        - returned_to_channel: Whether price came back inside
        - bounces_after_return: Number of false breaks before permanent exit
        - channel_continued: Whether pattern resumed after return
    """
    channel = detected.channel
    end_idx = detected.end_idx
    tf = detected.tf

    # Validate channel has required attributes
    if channel.slope is None or channel.intercept is None or channel.std_dev is None:
        return ChannelLabels(
            duration_bars=0,
            break_direction=int(BreakDirection.UP),
            next_channel_direction=int(NewChannelDirection.SIDEWAYS),
            permanent_break=False,
            timeframe=tf,
            duration_valid=False,
            direction_valid=False,
            next_channel_valid=False,
            break_scan_valid=False
        )

    # Calculate available forward data
    forward_start = end_idx + 1
    forward_end = min(end_idx + max_scan + 1, len(resampled_df))

    # Check if we have any forward data
    if forward_start >= len(resampled_df) or forward_end <= forward_start:
        return ChannelLabels(
            duration_bars=0,
            break_direction=int(BreakDirection.UP),
            next_channel_direction=int(NewChannelDirection.SIDEWAYS),
            permanent_break=False,
            timeframe=tf,
            duration_valid=False,
            direction_valid=False,
            next_channel_valid=False,
            break_scan_valid=False
        )

    # Extract forward price arrays for break scanner
    forward_slice = resampled_df.iloc[forward_start:forward_end]
    forward_high = forward_slice['high'].values.astype(np.float64)
    forward_low = forward_slice['low'].values.astype(np.float64)
    forward_close = forward_slice['close'].values.astype(np.float64)

    # Use sophisticated break scanner
    try:
        result: BreakResult = scan_for_break(
            channel=channel,
            forward_high=forward_high,
            forward_low=forward_low,
            forward_close=forward_close,
            max_scan_bars=max_scan,
            return_threshold_bars=10  # Bars price must stay outside to be "permanent"
        )
    except InsufficientDataError:
        # Not enough data to scan - return invalid labels
        return ChannelLabels(
            duration_bars=0,
            break_direction=int(BreakDirection.UP),
            next_channel_direction=int(NewChannelDirection.SIDEWAYS),
            permanent_break=False,
            timeframe=tf,
            duration_valid=False,
            direction_valid=False,
            next_channel_valid=False,
            break_scan_valid=False
        )

    # Map BreakResult to ChannelLabels fields
    if not result.break_detected:
        # No break found within scan limit
        return ChannelLabels(
            duration_bars=max_scan,
            break_direction=int(BreakDirection.UP),  # Default
            next_channel_direction=int(NewChannelDirection.SIDEWAYS),
            permanent_break=False,
            timeframe=tf,
            # FIRST break scan fields - no break detected
            bars_to_first_break=max_scan,
            break_magnitude=0.0,
            returned_to_channel=False,
            bounces_after_return=0,
            channel_continued=True,  # No break means channel continued
            # PERMANENT break fields - no break detected
            permanent_break_direction=-1,  # -1 = none
            permanent_break_magnitude=0.0,
            bars_to_permanent_break=-1,
            # Validity flags
            duration_valid=False,
            direction_valid=False,
            next_channel_valid=False,
            break_scan_valid=True  # Scan was performed, just no break found
        )

    # bounces_after_return: Use false_break_count which counts all temporary exits
    # that returned. If the first break returned, bounces = false_break_count - 1
    # (since first break itself counts as one). If first break was permanent,
    # all false breaks happened during the scan.
    if result.is_false_break:
        # First break returned, so subsequent false breaks are "bounces after return"
        bounces_after_return = max(0, result.false_break_count - 1)
    else:
        # First break was permanent - any false breaks came before final exit
        bounces_after_return = result.false_break_count

    # channel_continued: True if price returned to channel (is_false_break)
    # The "channel pattern continuing" means the channel held and price stayed
    # or returned - essentially the inverse of is_permanent
    channel_continued = result.is_false_break

    # Use provided next_channel_direction if available (from hybrid method),
    # otherwise default to SIDEWAYS (unknown)
    if next_channel_direction is not None:
        next_channel_dir = next_channel_direction
        next_channel_valid_flag = True
    else:
        next_channel_dir = int(NewChannelDirection.SIDEWAYS)
        next_channel_valid_flag = False

    return ChannelLabels(
        # Core label values (duration_bars uses break_bar for consistency)
        duration_bars=result.break_bar,
        break_direction=result.break_direction,
        next_channel_direction=next_channel_dir,
        permanent_break=result.is_permanent,
        timeframe=tf,
        # FIRST break scan fields from BreakResult
        bars_to_first_break=result.break_bar,
        break_magnitude=result.break_magnitude,
        returned_to_channel=result.is_false_break,  # is_false_break means it returned
        bounces_after_return=bounces_after_return,
        channel_continued=channel_continued,
        # PERMANENT break fields from BreakResult
        permanent_break_direction=result.permanent_break_direction,
        permanent_break_magnitude=result.permanent_break_magnitude,
        bars_to_permanent_break=result.permanent_break_bar,
        # Validity flags
        duration_valid=True,
        direction_valid=True,
        next_channel_valid=next_channel_valid_flag,
        break_scan_valid=True
    )


def generate_all_labels(
    channel_map: ChannelMap,
    resampled_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    labeling_method: str = "next_channel",
    progress_callback=None,
    verbose: bool = True
) -> LabeledChannelMap:
    """
    PASS 2: Generate labels for all channels in the map.

    Supports three labeling methods:
    - "next_channel": Determines break by looking at next channel's start price
      (original method, works without resampled_dfs). Uses next channel to infer
      break_direction and gets next_channel_direction from next channel's direction.
    - "forward_scan": Scans forward bars to find first close outside channel bounds
      (requires resampled_dfs, uses TF_MAX_SCAN for scan limits). Provides accurate
      bars_to_first_break, break_magnitude, returned_to_channel, etc.
    - "hybrid": Best of both - uses forward_scan for break timing/features
      (bars_to_first_break, break_magnitude, permanent_break, etc.) and uses
      next_channel lookup for next_channel_direction. Requires resampled_dfs.

    Args:
        channel_map: The channel map from detect_all_channels()
        resampled_dfs: Optional dict of resampled DataFrames keyed by timeframe.
                       Required for "forward_scan" and "hybrid" methods.
        labeling_method: "next_channel" (default), "forward_scan", or "hybrid"
        progress_callback: Optional callback(tf, window, pct) for progress updates
        verbose: If True, print detailed progress logging

    Returns:
        LabeledChannelMap: {(tf, window): [LabeledChannel, ...]}
    """
    # Validate labeling_method
    valid_methods = ("next_channel", "forward_scan", "hybrid")
    if labeling_method not in valid_methods:
        raise ValueError(f"labeling_method must be one of {valid_methods}, got '{labeling_method}'")

    # If forward_scan or hybrid requested but no resampled_dfs, fall back to next_channel
    if labeling_method in ("forward_scan", "hybrid") and resampled_dfs is None:
        if verbose:
            print(f"[PASS 2] WARNING: {labeling_method} requested but resampled_dfs is None, "
                  "falling back to next_channel method")
        labeling_method = "next_channel"

    labeled_map: LabeledChannelMap = {}

    total_keys = len(channel_map)
    key_idx = 0

    # Calculate total channels to label for verbose output
    total_channels = sum(len(channels) for channels in channel_map.values())

    # Count unique TFs and windows
    unique_tfs = set(tf for tf, _ in channel_map.keys())
    unique_windows = set(window for _, window in channel_map.keys())

    if verbose:
        print(f"[PASS 2] Starting label generation (method={labeling_method}):")
        print(f"         TF/window combinations to process: {total_keys}")
        print(f"         Unique TFs: {len(unique_tfs)}, Unique windows: {len(unique_windows)}")
        print(f"         Total channels to label: {total_channels:,}")
        print()

    channels_labeled = 0
    valid_labels_count = 0

    for (tf, window), channels in channel_map.items():
        labeled_channels: List[LabeledChannel] = []
        num_channels = len(channels)

        if verbose:
            pct = (key_idx / total_keys) * 100 if total_keys > 0 else 0
            print(f"[PASS 2] Processing TF={tf}, window={window} ({key_idx + 1}/{total_keys}, {pct:.1f}%) - {num_channels} channels to label")

        # Get resampled df and max_scan for forward_scan method
        resampled_df = resampled_dfs.get(tf) if resampled_dfs else None
        max_scan = TF_MAX_SCAN.get(tf, 500)

        for idx, detected in enumerate(channels):
            # Choose labeling method
            if labeling_method == "hybrid" and resampled_df is not None:
                # Hybrid: Get next_channel_direction from map lookup, then use forward_scan
                # for break timing features
                next_channel_dir = None
                next_idx = -1
                if idx + 1 < len(channels):
                    next_channel = channels[idx + 1]
                    next_channel_dir = next_channel.direction  # 0=BEAR, 1=SIDEWAYS, 2=BULL
                    next_idx = idx + 1

                # Use forward_scan with next_channel_direction for best of both
                labels = label_channel_forward_scan(
                    detected, resampled_df, max_scan,
                    next_channel_direction=next_channel_dir
                )
            elif labeling_method == "forward_scan" and resampled_df is not None:
                labels = label_channel_forward_scan(detected, resampled_df, max_scan)
                next_idx = -1  # forward_scan doesn't determine next channel
            else:
                # next_channel method
                labels, next_idx = label_channel_from_map(channel_map, tf, window, idx)

            labeled = LabeledChannel(
                detected=detected,
                labels=labels,
                next_channel_idx=next_idx
            )
            labeled_channels.append(labeled)
            channels_labeled += 1

            if labels.direction_valid:
                valid_labels_count += 1

        labeled_map[(tf, window)] = labeled_channels

        key_idx += 1
        if progress_callback:
            progress_callback(tf, window, key_idx / total_keys)

    if verbose:
        print()
        print(f"[PASS 2] Complete:")
        print(f"         Total channels labeled: {channels_labeled:,}")
        if channels_labeled > 0:
            print(f"         Valid direction labels: {valid_labels_count:,} ({100 * valid_labels_count / channels_labeled:.1f}% of total)")
        else:
            print("         No channels labeled")

    return labeled_map


# =============================================================================
# LOOKUP: Get Labels for Position
# =============================================================================

def get_labels_for_position(
    labeled_map: LabeledChannelMap,
    df: pd.DataFrame,
    position_idx: int,
    tf: str,
    window: int,
    verbose: bool = False
) -> Optional[ChannelLabels]:
    """
    Lazy O(log N) lookup of labels for a specific position in the dataset.

    Uses binary search to efficiently find the channel that contains or ends at
    position_idx and returns its pre-computed labels. This is the bridge between
    the two-pass labeling system and the training pipeline which needs labels
    for specific sample positions.

    The function handles three cases:
    1. Position falls within a channel's window -> returns that channel's labels
    2. Position is between channels -> returns the most recent channel's labels
    3. Position is before all channels or no channels exist -> returns None

    Time Complexity: O(log N) where N is the number of channels for this TF/window

    Args:
        labeled_map: The labeled channel map from generate_all_labels().
                     Keys are (tf, window) tuples, values are lists of LabeledChannel.
        df: The original 5min DataFrame with DatetimeIndex (for timestamp lookups).
        position_idx: Index in the 5min data to look up labels for.
        tf: Timeframe to look up (e.g., '5min', '15min', '1h', '1d').
        window: Window size to look up (e.g., 10, 20, 50, 100).
        verbose: If True, print debug logging for troubleshooting lookups.
                 Default is False for production use.

    Returns:
        ChannelLabels if a matching channel is found, None otherwise.
        Returns None (never raises) for edge cases:
        - No channels exist for this TF/window combination
        - Position is before any detected channels
        - Position index is out of bounds for the DataFrame
        - DataFrame is empty or has invalid index

    Example:
        >>> labels = get_labels_for_position(labeled_map, df, 50000, '1h', 20, verbose=True)
        >>> if labels is not None:
        ...     print(f"Duration: {labels.duration_bars}, Break: {labels.break_direction}")
    """
    key = (tf, window)
    labeled_channels = labeled_map.get(key, [])

    # Edge case: No channels for this TF/window combination
    if not labeled_channels:
        if verbose:
            print(f"[get_labels_for_position] No channels found for key={key}")
        return None

    # Edge case: Invalid position index
    if position_idx < 0 or position_idx >= len(df):
        if verbose:
            print(f"[get_labels_for_position] Position {position_idx} out of bounds "
                  f"(df length={len(df)})")
        return None

    # Edge case: Empty DataFrame
    if len(df) == 0:
        if verbose:
            print(f"[get_labels_for_position] DataFrame is empty")
        return None

    try:
        position_ts = df.index[position_idx]
    except Exception as e:
        if verbose:
            print(f"[get_labels_for_position] Failed to get timestamp at index {position_idx}: {e}")
        return None

    if verbose:
        print(f"[get_labels_for_position] Looking up: tf={tf}, window={window}, "
              f"position_idx={position_idx}, timestamp={position_ts}")
        print(f"[get_labels_for_position] Searching {len(labeled_channels)} channels...")

    # Edge case: Position is before all channels
    first_channel = labeled_channels[0]
    if position_ts < first_channel.detected.start_timestamp:
        if verbose:
            print(f"[get_labels_for_position] Position {position_ts} is before first channel "
                  f"(starts at {first_channel.detected.start_timestamp})")
        return None

    # Binary search for channel containing this timestamp
    # Channels are sorted by start_idx, which corresponds to timestamps
    left, right = 0, len(labeled_channels) - 1
    iterations = 0

    while left <= right:
        iterations += 1
        mid = (left + right) // 2
        lc = labeled_channels[mid]

        if verbose:
            print(f"[get_labels_for_position] Binary search iter {iterations}: "
                  f"left={left}, right={right}, mid={mid}, "
                  f"channel range=[{lc.detected.start_timestamp}, {lc.detected.end_timestamp}]")

        # Check if position_ts is within this channel's window
        if lc.detected.start_timestamp <= position_ts <= lc.detected.end_timestamp:
            if verbose:
                print(f"[get_labels_for_position] FOUND: Position within channel {mid} "
                      f"(duration={lc.labels.duration_bars}, "
                      f"break_dir={lc.labels.break_direction}, "
                      f"valid={lc.labels.direction_valid})")
            return lc.labels
        elif position_ts < lc.detected.start_timestamp:
            right = mid - 1
        else:
            left = mid + 1

    if verbose:
        print(f"[get_labels_for_position] No exact match after {iterations} iterations. "
              f"Searching for most recent channel...")

    # No exact match - return the most recent channel that ended before position
    # This handles cases where we're between channels (position is after one channel
    # ended but before the next one starts)
    #
    # Edge case: Position is after all channels
    # In this case, return the last channel's labels
    for idx, lc in enumerate(reversed(labeled_channels)):
        if lc.detected.end_timestamp <= position_ts:
            if verbose:
                actual_idx = len(labeled_channels) - 1 - idx
                print(f"[get_labels_for_position] FOUND: Most recent channel {actual_idx} "
                      f"ended at {lc.detected.end_timestamp} "
                      f"(duration={lc.labels.duration_bars}, "
                      f"break_dir={lc.labels.break_direction}, "
                      f"valid={lc.labels.direction_valid})")
            return lc.labels

    # This should not happen if the first channel check passed, but handle gracefully
    if verbose:
        print(f"[get_labels_for_position] No matching channel found (unexpected)")
    return None


# =============================================================================
# Statistics Utilities
# =============================================================================

def channel_map_stats(channel_map: ChannelMap) -> Dict:
    """
    Get statistics about a channel map.

    Useful for debugging and understanding the detected channels.

    Args:
        channel_map: The channel map from detect_all_channels()

    Returns:
        Dict with statistics
    """
    stats = {
        'total_channels': 0,
        'channels_per_tf': {},
        'channels_per_window': {},
        'direction_counts': {0: 0, 1: 0, 2: 0},  # BEAR, SIDEWAYS, BULL
        'avg_channels_per_combo': 0.0,
    }

    for (tf, window), channels in channel_map.items():
        count = len(channels)
        stats['total_channels'] += count

        # Per TF
        if tf not in stats['channels_per_tf']:
            stats['channels_per_tf'][tf] = 0
        stats['channels_per_tf'][tf] += count

        # Per window
        if window not in stats['channels_per_window']:
            stats['channels_per_window'][window] = 0
        stats['channels_per_window'][window] += count

        # Direction counts
        for ch in channels:
            stats['direction_counts'][ch.direction] += 1

    n_combos = len(channel_map)
    if n_combos > 0:
        stats['avg_channels_per_combo'] = stats['total_channels'] / n_combos

    return stats


def labeled_map_stats(labeled_map: LabeledChannelMap) -> Dict:
    """
    Get statistics about a labeled channel map.

    Args:
        labeled_map: The labeled channel map from generate_all_labels()

    Returns:
        Dict with label statistics
    """
    stats = {
        'total_labeled': 0,
        'valid_direction_count': 0,
        'valid_duration_count': 0,
        'break_up_count': 0,
        'break_down_count': 0,
        'next_bull_count': 0,
        'next_bear_count': 0,
        'next_sideways_count': 0,
        'avg_duration_bars': 0.0,
    }

    durations = []

    for (tf, window), labeled_channels in labeled_map.items():
        for lc in labeled_channels:
            stats['total_labeled'] += 1
            labels = lc.labels

            if labels.direction_valid:
                stats['valid_direction_count'] += 1

                if labels.break_direction == BreakDirection.UP:
                    stats['break_up_count'] += 1
                else:
                    stats['break_down_count'] += 1

                if labels.next_channel_direction == NewChannelDirection.BULL:
                    stats['next_bull_count'] += 1
                elif labels.next_channel_direction == NewChannelDirection.BEAR:
                    stats['next_bear_count'] += 1
                else:
                    stats['next_sideways_count'] += 1

            if labels.duration_valid:
                stats['valid_duration_count'] += 1
                durations.append(labels.duration_bars)

    if durations:
        stats['avg_duration_bars'] = sum(durations) / len(durations)

    return stats


# =============================================================================
# CROSS-CORRELATION LABEL COMPUTATION
# =============================================================================

def compute_cross_correlation_labels(
    tsla_labels: ChannelLabels,
    spy_labels: ChannelLabels,
    tf: str
) -> CrossCorrelationLabels:
    """
    Compute cross-correlation labels comparing TSLA and SPY break behavior.

    This function analyzes the relationship between TSLA and SPY channel breaks
    to identify alignment patterns that may be predictive of future movements.

    The function handles cases where one or both assets have invalid labels
    gracefully by returning a CrossCorrelationLabels with cross_valid=False.

    Args:
        tsla_labels: ChannelLabels for TSLA at this timeframe/window
        spy_labels: ChannelLabels for SPY at this timeframe/window
        tf: Timeframe string (e.g., '5min', '1h', 'daily') - used for context

    Returns:
        CrossCorrelationLabels with computed cross-asset comparison metrics.
        If either TSLA or SPY labels are invalid, returns default labels with
        cross_valid=False.

    Example:
        >>> tsla = ChannelLabels(break_direction=1, break_scan_valid=True, ...)
        >>> spy = ChannelLabels(break_direction=1, break_scan_valid=True, ...)
        >>> cross = compute_cross_correlation_labels(tsla, spy, '1h')
        >>> print(cross.break_direction_aligned)  # True if both broke same way
    """
    # Check if both labels have valid break scan data
    tsla_valid = tsla_labels.break_scan_valid if tsla_labels else False
    spy_valid = spy_labels.break_scan_valid if spy_labels else False

    # If either is invalid, return default invalid cross-correlation labels
    if not tsla_valid or not spy_valid:
        return CrossCorrelationLabels(cross_valid=False)

    # Both have valid break scan data - compute cross-correlation metrics

    # 1. Break direction alignment
    # Compare break_direction (from break scan) for both assets
    break_direction_aligned = (
        tsla_labels.break_direction == spy_labels.break_direction
    )

    # 2. Which asset broke first?
    # Compare bars_to_first_break - lower value means broke first
    tsla_bars = tsla_labels.bars_to_first_break
    spy_bars = spy_labels.bars_to_first_break

    tsla_broke_first = tsla_bars < spy_bars
    spy_broke_first = spy_bars < tsla_bars
    # If equal, neither is marked as first

    # 3. Break lag (absolute difference in bars)
    break_lag_bars = abs(tsla_bars - spy_bars)

    # 4. Magnitude spread (difference in break magnitudes)
    # Positive means TSLA broke with more magnitude
    magnitude_spread = tsla_labels.break_magnitude - spy_labels.break_magnitude

    # 5. Return behavior comparison
    tsla_returned = tsla_labels.returned_to_channel
    spy_returned = spy_labels.returned_to_channel

    both_returned = tsla_returned and spy_returned
    both_permanent = (not tsla_returned) and (not spy_returned)

    # Return pattern aligned if both returned or both didn't
    return_pattern_aligned = both_returned or both_permanent

    # 6. Continuation alignment
    # Check if both assets' channels continued after return (or both didn't)
    tsla_continued = tsla_labels.channel_continued
    spy_continued = spy_labels.channel_continued

    continuation_aligned = tsla_continued == spy_continued

    # ==========================================================================
    # PERMANENT BREAK CROSS-CORRELATION
    # ==========================================================================

    # Check if both have valid permanent breaks (-1 means no permanent break)
    tsla_has_perm = tsla_labels.permanent_break_direction >= 0
    spy_has_perm = spy_labels.permanent_break_direction >= 0
    permanent_cross_valid = tsla_has_perm and spy_has_perm

    # 7. Permanent break direction alignment
    permanent_direction_aligned = False
    if permanent_cross_valid:
        permanent_direction_aligned = (
            tsla_labels.permanent_break_direction == spy_labels.permanent_break_direction
        )

    # 8. Which asset achieved permanent break first?
    tsla_perm_bars = tsla_labels.bars_to_permanent_break
    spy_perm_bars = spy_labels.bars_to_permanent_break

    tsla_permanent_first = False
    spy_permanent_first = False
    permanent_break_lag_bars = 0

    if permanent_cross_valid:
        tsla_permanent_first = tsla_perm_bars < spy_perm_bars
        spy_permanent_first = spy_perm_bars < tsla_perm_bars
        permanent_break_lag_bars = abs(tsla_perm_bars - spy_perm_bars)
    elif tsla_has_perm and not spy_has_perm:
        tsla_permanent_first = True
    elif spy_has_perm and not tsla_has_perm:
        spy_permanent_first = True

    # 9. Permanent break magnitude spread
    permanent_magnitude_spread = 0.0
    if permanent_cross_valid:
        permanent_magnitude_spread = (
            tsla_labels.permanent_break_magnitude - spy_labels.permanent_break_magnitude
        )

    # ==========================================================================
    # DIRECTION TRANSITION PATTERNS (first break vs permanent break)
    # ==========================================================================

    # 10. Direction divergence (did permanent direction differ from first direction?)
    tsla_direction_diverged = False
    spy_direction_diverged = False

    if tsla_has_perm:
        tsla_direction_diverged = (
            tsla_labels.permanent_break_direction != tsla_labels.break_direction
        )
    if spy_has_perm:
        spy_direction_diverged = (
            spy_labels.permanent_break_direction != spy_labels.break_direction
        )

    # Both changed direction from first to permanent
    both_direction_diverged = tsla_direction_diverged and spy_direction_diverged

    # Divergence pattern aligned (both diverged OR both didn't)
    direction_divergence_aligned = tsla_direction_diverged == spy_direction_diverged

    return CrossCorrelationLabels(
        # FIRST break cross-correlation
        break_direction_aligned=break_direction_aligned,
        tsla_broke_first=tsla_broke_first,
        spy_broke_first=spy_broke_first,
        break_lag_bars=break_lag_bars,
        magnitude_spread=magnitude_spread,
        # PERMANENT break cross-correlation
        permanent_direction_aligned=permanent_direction_aligned,
        tsla_permanent_first=tsla_permanent_first,
        spy_permanent_first=spy_permanent_first,
        permanent_break_lag_bars=permanent_break_lag_bars,
        permanent_magnitude_spread=permanent_magnitude_spread,
        # Direction transition patterns
        tsla_direction_diverged=tsla_direction_diverged,
        spy_direction_diverged=spy_direction_diverged,
        both_direction_diverged=both_direction_diverged,
        direction_divergence_aligned=direction_divergence_aligned,
        # Return/permanence patterns
        both_returned=both_returned,
        both_permanent=both_permanent,
        return_pattern_aligned=return_pattern_aligned,
        continuation_aligned=continuation_aligned,
        # Validity flags
        cross_valid=True,
        permanent_cross_valid=permanent_cross_valid
    )
