"""
Parallel Channel Scanner for v15.

Scans positions in parallel using ProcessPoolExecutor to detect channels,
extract features, and generate labels across the dataset.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from v7.core.channel import detect_channels_multi_window, select_best_channel
from v15.types import ChannelSample, STANDARD_WINDOWS
from v15.features import extract_features
from v15.labels import generate_labels_multi_window


# Chunk size for parallel processing
CHUNK_SIZE = 100


@dataclass
class ScanConfig:
    """Configuration for scanner workers."""
    step: int = 10
    warmup_bars: int = 32760
    forward_bars: int = 8000


def _process_single_position(
    idx: int,
    tsla_data: np.ndarray,
    tsla_index: np.ndarray,
    spy_data: np.ndarray,
    vix_data: np.ndarray,
    columns: List[str],
    vix_columns: List[str],
    forward_bars: int = 8000
) -> Optional[Dict[str, Any]]:
    """
    Process a single position and return sample data.

    Args:
        idx: Position index in the data
        tsla_data: TSLA OHLCV values as numpy array
        tsla_index: TSLA timestamps as numpy array
        spy_data: SPY OHLCV values as numpy array
        vix_data: VIX OHLCV values as numpy array
        columns: Column names for TSLA/SPY DataFrame reconstruction
        vix_columns: Column names for VIX DataFrame reconstruction
        forward_bars: Number of forward bars for label scanning

    Returns:
        Dictionary with sample data or None if no valid channel
    """
    # Reconstruct DataFrames from numpy arrays (slice up to idx for channel detection)
    tsla_slice = pd.DataFrame(
        tsla_data[:idx],
        index=pd.DatetimeIndex(tsla_index[:idx]),
        columns=columns
    )
    spy_slice = pd.DataFrame(
        spy_data[:idx],
        index=pd.DatetimeIndex(tsla_index[:idx]),
        columns=columns
    )
    vix_slice = pd.DataFrame(
        vix_data[:idx],
        index=pd.DatetimeIndex(tsla_index[:idx]),
        columns=vix_columns
    )

    # Detect channels at all window sizes
    channels = detect_channels_multi_window(tsla_slice, windows=STANDARD_WINDOWS)

    # Select best channel
    best_channel, best_window = select_best_channel(channels)

    # Skip if no valid channel
    if best_channel is None or not best_channel.valid:
        return None

    # Extract features for ALL valid windows
    features_per_window = {}
    for window, channel in channels.items():
        if channel.valid:
            features_per_window[window] = extract_features(
                tsla_df=tsla_slice,
                spy_df=spy_slice,
                vix_df=vix_slice,
                channel=channel,
                window=window,
                channels_by_window=channels
            )

    # For label generation, include forward data for break scanning
    # Use idx + forward_bars to include enough forward data
    forward_end = min(idx + forward_bars, len(tsla_data))
    tsla_with_forward = pd.DataFrame(
        tsla_data[:forward_end],
        index=pd.DatetimeIndex(tsla_index[:forward_end]),
        columns=columns
    )

    # Generate labels for all windows across all timeframes
    # Pass data with forward bars, channel_end_idx points to where channel ends
    labels = generate_labels_multi_window(
        df=tsla_with_forward,
        channels=channels,
        channel_end_idx_5min=idx - 1  # Last index where channel was detected
    )

    # Return serializable data (no complex objects)
    return {
        'idx': idx,
        'timestamp': tsla_index[idx - 1],
        'best_window': best_window,
        'features_per_window': features_per_window,
        'labels': labels,
        'channel_valid': {w: c.valid for w, c in channels.items()},
        'channel_bounce_count': {w: c.bounce_count for w, c in channels.items()},
        'channel_r_squared': {w: c.r_squared for w, c in channels.items()},
        'channel_direction': {w: int(c.direction) for w, c in channels.items()},
    }


def _process_chunk(
    chunk_indices: List[int],
    tsla_data: np.ndarray,
    tsla_index: np.ndarray,
    spy_data: np.ndarray,
    vix_data: np.ndarray,
    columns: List[str],
    vix_columns: List[str],
    forward_bars: int = 8000
) -> List[Optional[Dict[str, Any]]]:
    """
    Process a chunk of positions.

    Args:
        chunk_indices: List of position indices to process
        tsla_data: TSLA OHLCV values as numpy array
        tsla_index: TSLA timestamps as numpy array
        spy_data: SPY OHLCV values as numpy array
        vix_data: VIX OHLCV values as numpy array
        columns: Column names for TSLA/SPY DataFrame reconstruction
        vix_columns: Column names for VIX DataFrame reconstruction
        forward_bars: Number of forward bars for label scanning

    Returns:
        List of sample dictionaries (None for invalid positions)
    """
    results = []
    for idx in chunk_indices:
        result = _process_single_position(
            idx=idx,
            tsla_data=tsla_data,
            tsla_index=tsla_index,
            spy_data=spy_data,
            vix_data=vix_data,
            columns=columns,
            vix_columns=vix_columns,
            forward_bars=forward_bars
        )
        results.append(result)
    return results


def _dict_to_channel_sample(data: Dict[str, Any]) -> ChannelSample:
    """
    Convert a dictionary back to a ChannelSample object.

    Args:
        data: Dictionary with sample data

    Returns:
        ChannelSample object
    """
    return ChannelSample(
        timestamp=pd.Timestamp(data['timestamp']),
        channel_end_idx=data['idx'],
        channels={},  # Don't store full channel objects
        features_per_window=data['features_per_window'],
        labels_per_window=data['labels'],
        best_window=data['best_window']
    )


def scan_channels(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    step: int = 10,
    warmup_bars: int = 32760,
    forward_bars: int = 8000,
    workers: int = 4,
    progress: bool = True
) -> List[ChannelSample]:
    """
    Scan for channels across the dataset using parallel processing.

    Processes positions from warmup_bars to len(tsla) - forward_bars,
    detecting channels, extracting features, and generating labels.

    Args:
        tsla_df: TSLA OHLCV DataFrame with DatetimeIndex
        spy_df: SPY OHLCV DataFrame aligned to TSLA
        vix_df: VIX OHLCV DataFrame aligned to TSLA
        step: Number of bars between samples (default 10)
        warmup_bars: Minimum bars required before first sample (default 32760)
        forward_bars: Bars reserved for forward label scanning (default 8000)
        workers: Number of parallel workers (default 4)
        progress: Show progress bar (default True)

    Returns:
        List of ChannelSample objects with features and labels

    Example:
        >>> tsla, spy, vix = load_market_data('/path/to/data')
        >>> samples = scan_channels(tsla, spy, vix, workers=8)
        >>> print(f"Found {len(samples)} valid samples")
    """
    n_bars = len(tsla_df)

    # Calculate position range
    start_idx = warmup_bars
    end_idx = n_bars - forward_bars

    if start_idx >= end_idx:
        print(f"Error: Not enough data. Have {n_bars} bars, need at least "
              f"{warmup_bars + forward_bars} for warmup + forward scanning.")
        return []

    # Generate all position indices
    positions = list(range(start_idx, end_idx, step))
    total_positions = len(positions)

    print(f"Scanning {total_positions} positions from {start_idx} to {end_idx} (step={step})")
    print(f"Data range: {n_bars} bars, warmup={warmup_bars}, forward={forward_bars}")
    print(f"Using {workers} workers with chunk size {CHUNK_SIZE}")

    # Convert DataFrames to numpy arrays for efficient serialization
    columns = ['open', 'high', 'low', 'close', 'volume']
    tsla_data = tsla_df[columns].values
    tsla_index = tsla_df.index.values
    spy_data = spy_df[columns].values

    # VIX may have fewer columns (no volume), so handle separately
    vix_cols = [c for c in columns if c in vix_df.columns]
    vix_data = vix_df[vix_cols].values if vix_cols else np.zeros((len(vix_df), 4))

    # Split positions into chunks
    chunks = [
        positions[i:i + CHUNK_SIZE]
        for i in range(0, len(positions), CHUNK_SIZE)
    ]

    # Process chunks in parallel using spawn context
    samples = []
    valid_count = 0

    # Use spawn context for clean process isolation
    ctx = mp.get_context('spawn')

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        # Submit all chunks
        futures = {
            executor.submit(
                _process_chunk,
                chunk,
                tsla_data,
                tsla_index,
                spy_data,
                vix_data,
                columns,
                vix_cols,
                forward_bars
            ): i for i, chunk in enumerate(chunks)
        }

        # Collect results with progress bar
        if progress:
            pbar = tqdm(
                total=total_positions,
                desc="Scanning channels",
                unit="pos"
            )

        for future in as_completed(futures):
            chunk_results = future.result()

            for result in chunk_results:
                if result is not None:
                    sample = _dict_to_channel_sample(result)
                    samples.append(sample)
                    valid_count += 1

                if progress:
                    pbar.update(1)

        if progress:
            pbar.close()

    # Sort samples by index
    samples.sort(key=lambda s: s.channel_end_idx)

    # Print summary
    print(f"\nScan complete!")
    print(f"  Total positions scanned: {total_positions}")
    print(f"  Valid samples found: {valid_count}")
    print(f"  Skip rate: {100 * (total_positions - valid_count) / total_positions:.1f}%")

    return samples


def scan_channels_sequential(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    step: int = 10,
    warmup_bars: int = 32760,
    forward_bars: int = 8000,
    progress: bool = True
) -> List[ChannelSample]:
    """
    Sequential version of scan_channels for debugging or small datasets.

    Same interface as scan_channels but runs single-threaded.

    Args:
        tsla_df: TSLA OHLCV DataFrame with DatetimeIndex
        spy_df: SPY OHLCV DataFrame aligned to TSLA
        vix_df: VIX OHLCV DataFrame aligned to TSLA
        step: Number of bars between samples (default 10)
        warmup_bars: Minimum bars required before first sample (default 32760)
        forward_bars: Bars reserved for forward label scanning (default 8000)
        progress: Show progress bar (default True)

    Returns:
        List of ChannelSample objects with features and labels
    """
    n_bars = len(tsla_df)

    # Calculate position range
    start_idx = warmup_bars
    end_idx = n_bars - forward_bars

    if start_idx >= end_idx:
        print(f"Error: Not enough data. Have {n_bars} bars, need at least "
              f"{warmup_bars + forward_bars} for warmup + forward scanning.")
        return []

    # Generate all position indices
    positions = list(range(start_idx, end_idx, step))
    total_positions = len(positions)

    print(f"Scanning {total_positions} positions sequentially")

    samples = []
    valid_count = 0

    iterator = tqdm(positions, desc="Scanning", unit="pos") if progress else positions

    for idx in iterator:
        # Slice data up to idx
        tsla_slice = tsla_df.iloc[:idx]
        spy_slice = spy_df.iloc[:idx]
        vix_slice = vix_df.iloc[:idx]

        # Detect channels
        channels = detect_channels_multi_window(tsla_slice, windows=STANDARD_WINDOWS)

        # Select best channel
        best_channel, best_window = select_best_channel(channels)

        if best_channel is None or not best_channel.valid:
            continue

        # Extract features for ALL valid windows
        features_per_window = {}
        for window, channel in channels.items():
            if channel.valid:
                features_per_window[window] = extract_features(
                    tsla_df=tsla_slice,
                    spy_df=spy_slice,
                    vix_df=vix_slice,
                    channel=channel,
                    window=window,
                    channels_by_window=channels
                )

        # For label generation, include forward data for break scanning
        forward_end = min(idx + forward_bars, len(tsla_df))
        tsla_with_forward = tsla_df.iloc[:forward_end]

        # Generate labels
        labels = generate_labels_multi_window(
            df=tsla_with_forward,
            channels=channels,
            channel_end_idx_5min=idx - 1
        )

        # Create sample
        sample = ChannelSample(
            timestamp=tsla_df.index[idx - 1],
            channel_end_idx=idx,
            channels={},
            features_per_window=features_per_window,
            labels_per_window=labels,
            best_window=best_window
        )

        samples.append(sample)
        valid_count += 1

    # Print summary
    print(f"\nScan complete!")
    print(f"  Total positions scanned: {total_positions}")
    print(f"  Valid samples found: {valid_count}")
    print(f"  Skip rate: {100 * (total_positions - valid_count) / total_positions:.1f}%")

    return samples


if __name__ == "__main__":
    # Simple test
    from v15.data import load_market_data

    print("Loading market data...")
    tsla, spy, vix = load_market_data("data")
    print(f"Loaded {len(tsla)} bars")

    print("\nRunning parallel scan...")
    samples = scan_channels(
        tsla, spy, vix,
        step=100,  # Use larger step for quick test
        workers=4,
        progress=True
    )

    print(f"\nGenerated {len(samples)} samples")
    if samples:
        print(f"First sample: {samples[0].timestamp}")
        print(f"Last sample: {samples[-1].timestamp}")
