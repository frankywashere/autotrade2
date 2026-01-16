"""
Parallel Channel Scanner for v15.

REWRITTEN to use the new v15 module system:
- data.loader for data loading
- data.resampler for partial bars (keeps incomplete bars with metadata)
- features.extractor for feature extraction
- LOUD failures - no silent try/except
- Validates all outputs before storing

Generates samples with tf_features (8,665 features) compatible with the new model.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# V15 module imports
from v15.types import ChannelSample, STANDARD_WINDOWS, TIMEFRAMES
from v15.config import (
    SCANNER_CONFIG, TOTAL_FEATURES, FEATURE_COUNTS,
    TIMEFRAMES as CONFIG_TIMEFRAMES
)
from v15.exceptions import (
    DataLoadError, FeatureExtractionError, ValidationError,
    InvalidFeatureError, ResamplingError, ChannelDetectionError
)
from v15.data import load_market_data, resample_with_partial
from v15.features.tf_extractor import extract_all_tf_features, get_tf_feature_count
from v15.features.validation import validate_features
from v15.labels import generate_labels_multi_window

# V7 channel detection (still needed)
from v7.core.channel import detect_channels_multi_window, select_best_channel


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
        # Map TF names to pandas resample rules
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
            '3month': '3MS',
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

    Total: 33 features (11 TFs * 3 features)

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
# Single Position Processing
# =============================================================================

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

    NO SILENT FAILURES - raises exceptions that propagate up.

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

    Raises:
        FeatureExtractionError: If feature extraction fails
        ValidationError: If validation fails
        ChannelDetectionError: If channel detection fails unexpectedly
    """
    # Reconstruct DataFrames from numpy arrays
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

    # Detect channels at all window sizes - NO silent catch
    channels = detect_channels_multi_window(tsla_slice, windows=STANDARD_WINDOWS)

    # Select best channel
    best_channel, best_window = select_best_channel(channels)

    # Skip if no valid channel (this is expected, not an error)
    if best_channel is None or not best_channel.valid:
        return None

    timestamp = pd.Timestamp(tsla_index[idx - 1])

    # Extract TF-aware features with resampling metadata
    # This extracts ~8,632 base features
    tf_features = extract_all_tf_features(
        tsla_df=tsla_slice,
        spy_df=spy_slice,
        vix_df=vix_slice,
        timestamp=timestamp
    )

    # Resample and get bar metadata for partial bar features
    _, bar_metadata = resample_all_timeframes(tsla_slice, idx)

    # Add bar metadata features (33 features)
    bar_meta_features = extract_bar_metadata_features(bar_metadata)
    tf_features.update(bar_meta_features)

    # VALIDATE features - LOUD failure on any invalid
    tf_features = validate_sample_features(tf_features, timestamp, idx)

    # For label generation, include forward data
    forward_end = min(idx + forward_bars, len(tsla_data))
    tsla_with_forward = pd.DataFrame(
        tsla_data[:forward_end],
        index=pd.DatetimeIndex(tsla_index[:forward_end]),
        columns=columns
    )

    # Generate labels for all windows across all timeframes
    labels = generate_labels_multi_window(
        df=tsla_with_forward,
        channels=channels,
        channel_end_idx_5min=idx - 1
    )

    # Return serializable data
    return {
        'idx': idx,
        'timestamp': tsla_index[idx - 1],
        'best_window': best_window,
        'tf_features': tf_features,
        'labels': labels,
        'bar_metadata': bar_metadata,
        'channel_valid': {w: c.valid for w, c in channels.items()},
        'channel_bounce_count': {w: c.bounce_count for w, c in channels.items()},
        'channel_r_squared': {w: c.r_squared for w, c in channels.items()},
        'channel_direction': {w: int(c.direction) for w, c in channels.items()},
        'n_features': len(tf_features),
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
) -> List[Dict[str, Any]]:
    """
    Process a chunk of positions.

    Collects results and errors for the entire chunk.

    Args:
        chunk_indices: List of position indices to process
        [other args same as _process_single_position]

    Returns:
        List of dicts with 'result' or 'error' keys
    """
    results = []

    for idx in chunk_indices:
        try:
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
            results.append({'result': result, 'error': None, 'idx': idx})
        except Exception as e:
            # Capture error with context - LOUD but don't crash entire chunk
            results.append({
                'result': None,
                'error': f"Error at idx={idx}: {type(e).__name__}: {str(e)}",
                'idx': idx
            })

    return results


# =============================================================================
# Dictionary to ChannelSample Conversion
# =============================================================================

def _dict_to_channel_sample(data: Dict[str, Any]) -> ChannelSample:
    """
    Convert a dictionary to a ChannelSample object.

    Args:
        data: Dictionary with sample data

    Returns:
        ChannelSample object with tf_features
    """
    return ChannelSample(
        timestamp=pd.Timestamp(data['timestamp']),
        channel_end_idx=data['idx'],
        tf_features=data['tf_features'],
        labels_per_window=data['labels'],
        bar_metadata=data.get('bar_metadata', {}),
        best_window=data['best_window']
    )


# =============================================================================
# Main Scanner Functions
# =============================================================================

def scan_channels(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    step: int = 10,
    warmup_bars: int = 32760,
    forward_bars: int = 8000,
    workers: int = 4,
    progress: bool = True,
    strict: bool = True
) -> List[ChannelSample]:
    """
    Scan for channels across the dataset using parallel processing.

    Uses the new v15 module system:
    - resample_with_partial for partial bar handling
    - extract_all_tf_features for 8,665 feature extraction
    - Validates all features before storing

    Args:
        tsla_df: TSLA OHLCV DataFrame with DatetimeIndex
        spy_df: SPY OHLCV DataFrame aligned to TSLA
        vix_df: VIX OHLCV DataFrame aligned to TSLA
        step: Number of bars between samples (default 10)
        warmup_bars: Minimum bars required before first sample (default 32760)
        forward_bars: Bars reserved for forward label scanning (default 8000)
        workers: Number of parallel workers (default 4)
        progress: Show progress bar (default True)
        strict: If True, raise on any errors. If False, skip failed samples (default True)

    Returns:
        List of ChannelSample objects with tf_features and labels

    Raises:
        DataLoadError: If data validation fails
        ValidationError: If strict=True and any sample fails validation
    """
    n_bars = len(tsla_df)

    # Calculate position range
    start_idx = warmup_bars
    end_idx = n_bars - forward_bars

    if start_idx >= end_idx:
        raise DataLoadError(
            f"Not enough data. Have {n_bars} bars, need at least "
            f"{warmup_bars + forward_bars} for warmup ({warmup_bars}) + "
            f"forward scanning ({forward_bars})."
        )

    # Generate all position indices
    positions = list(range(start_idx, end_idx, step))
    total_positions = len(positions)
    chunk_size = ScanConfig.chunk_size

    print(f"Scanning {total_positions} positions from {start_idx} to {end_idx} (step={step})")
    print(f"Data range: {n_bars} bars, warmup={warmup_bars}, forward={forward_bars}")
    print(f"Using {workers} workers with chunk size {chunk_size}")
    print(f"Expected features per sample: {EXPECTED_FEATURE_COUNT}")

    # Convert DataFrames to numpy arrays for efficient serialization
    columns = ['open', 'high', 'low', 'close', 'volume']
    tsla_data = tsla_df[columns].values
    tsla_index = tsla_df.index.values
    spy_data = spy_df[columns].values

    # VIX may have fewer columns
    vix_cols = [c for c in columns if c in vix_df.columns]
    vix_data = vix_df[vix_cols].values if vix_cols else np.zeros((len(vix_df), 4))

    # Split positions into chunks
    chunks = [
        positions[i:i + chunk_size]
        for i in range(0, len(positions), chunk_size)
    ]

    # Process chunks in parallel
    samples = []
    errors = []
    valid_count = 0
    skipped_count = 0

    # Use spawn context for clean process isolation
    ctx = mp.get_context('spawn')

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
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

        if progress:
            pbar = tqdm(total=total_positions, desc="Scanning channels", unit="pos")

        for future in as_completed(futures):
            chunk_results = future.result()

            for item in chunk_results:
                if progress:
                    pbar.update(1)

                if item['error'] is not None:
                    # LOUD error handling
                    errors.append(item['error'])
                    if strict:
                        if progress:
                            pbar.close()
                        raise ValidationError(item['error'])
                    continue

                result = item['result']
                if result is None:
                    skipped_count += 1
                    continue

                # Validate feature count
                n_features = result.get('n_features', 0)
                if n_features < 1000:  # Sanity check - should be ~8,665
                    error_msg = (
                        f"Too few features at idx={item['idx']}: "
                        f"got {n_features}, expected ~{EXPECTED_FEATURE_COUNT}"
                    )
                    errors.append(error_msg)
                    if strict:
                        if progress:
                            pbar.close()
                        raise ValidationError(error_msg)
                    continue

                sample = _dict_to_channel_sample(result)
                samples.append(sample)
                valid_count += 1

        if progress:
            pbar.close()

    # Sort samples by index
    samples.sort(key=lambda s: s.channel_end_idx)

    # Print summary
    print(f"\nScan complete!")
    print(f"  Total positions scanned: {total_positions}")
    print(f"  Valid samples found: {valid_count}")
    print(f"  Skipped (no channel): {skipped_count}")
    print(f"  Errors: {len(errors)}")

    if valid_count > 0:
        avg_features = sum(len(s.tf_features) for s in samples) / valid_count
        print(f"  Average features per sample: {avg_features:.0f}")

    skip_rate = 100 * (total_positions - valid_count) / total_positions
    print(f"  Skip rate: {skip_rate:.1f}%")

    if errors and not strict:
        print(f"\nWarning: {len(errors)} errors occurred (strict=False)")
        for i, err in enumerate(errors[:5]):
            print(f"  [{i+1}] {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    return samples


def scan_channels_sequential(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    step: int = 10,
    warmup_bars: int = 32760,
    forward_bars: int = 8000,
    progress: bool = True,
    strict: bool = True
) -> List[ChannelSample]:
    """
    Sequential version of scan_channels for debugging.

    Same interface as scan_channels but runs single-threaded.
    Useful for debugging feature extraction issues.

    Args:
        [same as scan_channels]

    Returns:
        List of ChannelSample objects with tf_features and labels

    Raises:
        FeatureExtractionError: If feature extraction fails
        ValidationError: If validation fails
    """
    n_bars = len(tsla_df)

    start_idx = warmup_bars
    end_idx = n_bars - forward_bars

    if start_idx >= end_idx:
        raise DataLoadError(
            f"Not enough data. Have {n_bars} bars, need at least "
            f"{warmup_bars + forward_bars}."
        )

    positions = list(range(start_idx, end_idx, step))
    total_positions = len(positions)

    print(f"Scanning {total_positions} positions sequentially")
    print(f"Expected features per sample: {EXPECTED_FEATURE_COUNT}")

    samples = []
    errors = []
    valid_count = 0
    skipped_count = 0

    iterator = tqdm(positions, desc="Scanning", unit="pos") if progress else positions

    for idx in iterator:
        try:
            # Slice data up to idx
            tsla_slice = tsla_df.iloc[:idx]
            spy_slice = spy_df.iloc[:idx]
            vix_slice = vix_df.iloc[:idx]

            # Detect channels
            channels = detect_channels_multi_window(tsla_slice, windows=STANDARD_WINDOWS)

            # Select best channel
            best_channel, best_window = select_best_channel(channels)

            if best_channel is None or not best_channel.valid:
                skipped_count += 1
                continue

            timestamp = tsla_df.index[idx - 1]

            # Extract features with partial bar support
            tf_features = extract_all_tf_features(
                tsla_df=tsla_slice,
                spy_df=spy_slice,
                vix_df=vix_slice,
                timestamp=timestamp
            )

            # Add bar metadata features
            _, bar_metadata = resample_all_timeframes(tsla_slice, idx)
            bar_meta_features = extract_bar_metadata_features(bar_metadata)
            tf_features.update(bar_meta_features)

            # VALIDATE - LOUD failure
            tf_features = validate_sample_features(tf_features, timestamp, idx)

            # Generate labels
            forward_end = min(idx + forward_bars, len(tsla_df))
            tsla_with_forward = tsla_df.iloc[:forward_end]

            labels = generate_labels_multi_window(
                df=tsla_with_forward,
                channels=channels,
                channel_end_idx_5min=idx - 1
            )

            # Create sample
            sample = ChannelSample(
                timestamp=timestamp,
                channel_end_idx=idx,
                channels={},
                tf_features=tf_features,
                features_per_window={},  # Legacy - empty
                labels_per_window=labels,
                best_window=best_window
            )

            samples.append(sample)
            valid_count += 1

        except Exception as e:
            error_msg = f"Error at idx={idx}: {type(e).__name__}: {str(e)}"
            errors.append(error_msg)
            if strict:
                raise ValidationError(error_msg) from e

    # Print summary
    print(f"\nScan complete!")
    print(f"  Total positions scanned: {total_positions}")
    print(f"  Valid samples found: {valid_count}")
    print(f"  Skipped (no channel): {skipped_count}")
    print(f"  Errors: {len(errors)}")

    if valid_count > 0:
        avg_features = sum(len(s.tf_features) for s in samples) / valid_count
        print(f"  Average features per sample: {avg_features:.0f}")

    skip_rate = 100 * (total_positions - valid_count) / total_positions
    print(f"  Skip rate: {skip_rate:.1f}%")

    return samples


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("V15 Channel Scanner - Using New Module System")
    print("=" * 60)

    print("\nLoading market data using v15.data.loader...")
    tsla, spy, vix = load_market_data("data")
    print(f"Loaded {len(tsla)} bars")
    print(f"Date range: {tsla.index[0]} to {tsla.index[-1]}")

    print(f"\nExpected feature count: {EXPECTED_FEATURE_COUNT}")
    print(f"  - TF-specific: {TOTAL_FEATURES - FEATURE_COUNTS['events_total']}")
    print(f"  - Events: {FEATURE_COUNTS['events_total']}")
    print(f"  - Bar metadata: {FEATURE_COUNTS['bar_metadata_per_tf'] * len(TIMEFRAMES)}")

    print("\nRunning parallel scan (step=100 for quick test)...")
    samples = scan_channels(
        tsla, spy, vix,
        step=100,  # Larger step for quick test
        workers=4,
        progress=True,
        strict=True  # LOUD failures
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
