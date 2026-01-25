"""
Two-Pass Channel Scanner for v15 - CHANNEL-END SAMPLING ARCHITECTURE.

SIMPLIFIED ARCHITECTURE:
1. PASS 1: Detect all channels across the dataset (detect_all_channels)
2. PASS 2: Compute labels at channel END positions (generate_all_labels)
3. SCAN: Iterate over detected channels. Each channel = ONE sample at its end_idx.

KEY PRINCIPLE - ONE SAMPLE PER CHANNEL:
- We iterate over DETECTED CHANNELS from the slim_map
- Each channel's end_idx IS the sample position
- Labels are PRECOMPUTED in Pass 2 and stored in SlimLabeledChannel
- For the PRIMARY channel, use its labels directly (no lookup needed)
- For OTHER TF/window combinations at same timestamp, do binary search lookup

The --step parameter controls CHANNEL DETECTION spacing in Pass 1.
Number of samples = number of valid detected channels.

Module dependencies:
- data.loader for data loading
- features.tf_extractor for feature extraction
- LOUD failures - no silent try/except
"""

import argparse
import gc
import multiprocessing as mp
import os
import pickle
import platform
import signal
import time
import traceback
from dataclasses import dataclass, replace
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Optional, Tuple, Any

import pandas as pd
from tqdm import tqdm

# V15 module imports
from v15.dtypes import ChannelSample, STANDARD_WINDOWS
from v15.config import (
    TOTAL_FEATURES,
    TIMEFRAMES,
    SCANNER_LOOKBACK_5MIN,
)
from v15.exceptions import (
    DataLoadError, FeatureExtractionError, ValidationError,
)
from v15.data import load_market_data
from v15.features.tf_extractor import extract_all_tf_features
from v15.features.validation import validate_features
from v15.labels import (
    # Two-pass labeling system imports
    detect_all_channels,
    generate_all_labels,
    ChannelMap,
    LabeledChannelMap,
)


# =============================================================================
# Multiprocessing Safety Guards
# =============================================================================

def _setup_multiprocessing():
    """Configure multiprocessing for cross-platform compatibility."""
    if platform.system() == 'Darwin':
        try:
            mp.set_start_method('spawn', force=False)
        except RuntimeError:
            pass


def _estimate_memory_per_worker(df_rows: int) -> float:
    """Estimate memory usage per worker in MB."""
    return (8 * 6 * df_rows * 3) / (1024 * 1024)


def _check_memory_availability(workers: int, df_rows: int):
    """Warn if memory might be insufficient for parallel processing."""
    try:
        import psutil
        mem_per_worker = _estimate_memory_per_worker(df_rows)
        total_needed = mem_per_worker * workers
        available = psutil.virtual_memory().available / (1024 * 1024)
        if total_needed > available * 0.8:
            print(f"WARNING: Estimated memory usage ({total_needed:.0f}MB) may exceed "
                  f"available memory ({available:.0f}MB). Consider reducing --workers.")
    except ImportError:
        pass


def _convert_df_to_pickle_safe(df: pd.DataFrame) -> Dict[str, Any]:
    """Convert DataFrame to pickle-safe format for worker processes."""
    return {
        'values': df.values.copy(),
        'columns': list(df.columns),
        'index': df.index.values.copy(),
        'index_name': df.index.name,
    }


def _reconstruct_df_from_pickle_safe(data: Dict[str, Any]) -> pd.DataFrame:
    """Reconstruct DataFrame from pickle-safe format."""
    return pd.DataFrame(
        data['values'],
        columns=data['columns'],
        index=pd.DatetimeIndex(data['index'], name=data['index_name'])
    )


@dataclass
class SlimLabeledChannel:
    """
    Memory-efficient channel with precomputed labels for worker processes.

    ARCHITECTURAL NOTE:
    - end_idx IS the sample position (in TF-space)
    - end_timestamp IS the sample timestamp
    - labels are PRECOMPUTED in Pass 2 - use them directly, no lookup needed
    """
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp
    start_idx: int = 0
    end_idx: int = 0               # THIS IS THE SAMPLE POSITION (in TF-space)
    # Channel regression parameters
    channel_slope: float = 0.0
    channel_intercept: float = 0.0
    channel_std_dev: float = 0.0
    channel_r_squared: float = 0.0
    channel_direction: int = -1    # 0=BEAR, 1=SIDEWAYS, 2=BULL
    channel_valid: bool = True
    channel_window: int = 0
    channel_bounce_count: int = 0
    tf: str = ""
    # PRECOMPUTED labels from Pass 2 - USE DIRECTLY
    labels: Any = None


def _create_slim_labeled_map(labeled_map: LabeledChannelMap) -> Dict[Tuple[str, int], List[SlimLabeledChannel]]:
    """
    Create a memory-efficient version of the labeled map for workers.

    Strips heavy numpy arrays (upper_line, lower_line, center_line) from channels.
    Memory reduction: ~100x per map (from GBs to MBs).
    """
    slim_map = {}
    for key, labeled_channels in labeled_map.items():
        tf, window = key
        slim_channels = []
        for lc in labeled_channels:
            channel_valid = False
            channel_bounce_count = 0
            channel_slope = 0.0
            channel_intercept = 0.0
            channel_std_dev = 0.0
            channel_r_squared = 0.0
            channel_direction = -1

            if lc.detected and lc.detected.channel:
                ch = lc.detected.channel
                channel_valid = getattr(ch, 'valid', False)
                channel_bounce_count = getattr(ch, 'bounce_count', 0)
                channel_slope = ch.slope if ch.slope is not None else 0.0
                channel_intercept = ch.intercept if ch.intercept is not None else 0.0
                channel_std_dev = ch.std_dev if ch.std_dev is not None else 0.0
                channel_r_squared = ch.r_squared if ch.r_squared is not None else 0.0
                channel_direction = int(ch.direction) if ch.direction is not None else -1

            slim_channels.append(SlimLabeledChannel(
                start_timestamp=lc.detected.start_timestamp,
                end_timestamp=lc.detected.end_timestamp,
                start_idx=lc.detected.start_idx,
                end_idx=lc.detected.end_idx,
                channel_slope=channel_slope,
                channel_intercept=channel_intercept,
                channel_std_dev=channel_std_dev,
                channel_r_squared=channel_r_squared,
                channel_direction=channel_direction,
                channel_valid=channel_valid,
                channel_window=window,
                channel_bounce_count=channel_bounce_count,
                tf=tf,
                labels=lc.labels  # Precomputed labels from Pass 2
            ))
        slim_map[key] = slim_channels
    return slim_map


# Global flag for graceful shutdown
_shutdown_requested = False

# Worker Process Globals (set via Pool initializer)
_WORKER_TSLA_DF: Optional[pd.DataFrame] = None
_WORKER_SPY_DF: Optional[pd.DataFrame] = None
_WORKER_VIX_DF: Optional[pd.DataFrame] = None
_WORKER_TSLA_SLIM_MAP: Optional[Dict] = None
_WORKER_SPY_SLIM_MAP: Optional[Dict] = None
_WORKER_TIMEFRAMES: Optional[List[str]] = None
_WORKER_WINDOWS: Optional[List[int]] = None


def _init_worker(tsla_data: Dict, spy_data: Dict, vix_data: Dict,
                 tsla_slim_map: Dict, spy_slim_map: Dict,
                 timeframes: List[str], windows: List[int]):
    """Initialize worker process with shared data."""
    global _WORKER_TSLA_DF, _WORKER_SPY_DF, _WORKER_VIX_DF
    global _WORKER_TSLA_SLIM_MAP, _WORKER_SPY_SLIM_MAP
    global _WORKER_TIMEFRAMES, _WORKER_WINDOWS

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    _WORKER_TSLA_DF = _reconstruct_df_from_pickle_safe(tsla_data)
    _WORKER_SPY_DF = _reconstruct_df_from_pickle_safe(spy_data)
    _WORKER_VIX_DF = _reconstruct_df_from_pickle_safe(vix_data)
    _WORKER_TSLA_SLIM_MAP = tsla_slim_map
    _WORKER_SPY_SLIM_MAP = spy_slim_map
    _WORKER_TIMEFRAMES = timeframes
    _WORKER_WINDOWS = windows


def _signal_handler(signum, frame):
    """Handle interrupt signals for graceful shutdown."""
    global _shutdown_requested
    if not _shutdown_requested:
        _shutdown_requested = True
        print("\n[INTERRUPT] Shutdown requested. Will stop after current batch...")


def _find_channel_at_timestamp(
    slim_map: Dict[Tuple[str, int], List[SlimLabeledChannel]],
    tf: str,
    window: int,
    timestamp: pd.Timestamp
) -> Optional[SlimLabeledChannel]:
    """
    Binary search to find channel ending at/before timestamp.

    Returns the SlimLabeledChannel (not just labels) so we can access all metadata.
    O(log N) complexity where N is number of channels for this (tf, window).
    """
    key = (tf, window)
    slim_channels = slim_map.get(key, [])

    if not slim_channels:
        return None

    left, right = 0, len(slim_channels) - 1
    found = None

    while left <= right:
        mid = (left + right) // 2
        lc = slim_channels[mid]

        if lc.end_timestamp <= timestamp:
            found = lc
            left = mid + 1
        else:
            right = mid - 1

    return found


def _process_channel_batch(channel_batch: List[Tuple[str, int, int]]) -> List[Dict[str, Any]]:
    """
    Process a batch of channels. Each channel produces ONE sample at channel.end_idx.

    SIMPLIFIED ARCHITECTURE:
    - channel_batch: List of (tf, window, channel_index) identifying channels
    - For each channel:
      1. Get channel from slim_map using (tf, window, index)
      2. Sample position = channel.end_timestamp
      3. Extract features at that position
      4. For PRIMARY channel (tf, window), use precomputed labels DIRECTLY
      5. For OTHER (tf, window) combinations, do binary search lookup

    Args:
        channel_batch: List of (tf, window, channel_idx) tuples

    Returns:
        List of result dicts with 'sample', 'error', or 'skipped' keys.
        Each successful result includes 'feature_extraction_time_ms' for timing stats.
    """
    try:
        tsla_df = _WORKER_TSLA_DF
        spy_df = _WORKER_SPY_DF
        vix_df = _WORKER_VIX_DF
        tsla_slim_map = _WORKER_TSLA_SLIM_MAP
        spy_slim_map = _WORKER_SPY_SLIM_MAP
        timeframes = _WORKER_TIMEFRAMES
        windows = _WORKER_WINDOWS

        from v15.features.tf_extractor import extract_all_tf_features
        from v15.scanner import validate_sample_features, EXPECTED_FEATURE_COUNT
        from v15.dtypes import ChannelSample

        results = []

        for primary_tf, primary_window, channel_idx in channel_batch:
            feature_extraction_time_ms = 0.0
            try:
                # Get the PRIMARY channel from slim_map
                key = (primary_tf, primary_window)
                slim_channels = tsla_slim_map.get(key, [])

                if channel_idx < 0 or channel_idx >= len(slim_channels):
                    results.append({'channel_key': (primary_tf, primary_window, channel_idx), 'error': "Invalid channel index"})
                    continue

                primary_channel = slim_channels[channel_idx]

                # Skip invalid channels
                if not primary_channel.channel_valid:
                    results.append({'channel_key': (primary_tf, primary_window, channel_idx), 'skipped': True, 'reason': 'invalid_channel'})
                    continue

                # Skip channels without valid labels
                if primary_channel.labels is None or not primary_channel.labels.direction_valid:
                    results.append({'channel_key': (primary_tf, primary_window, channel_idx), 'skipped': True, 'reason': 'invalid_labels'})
                    continue

                # SAMPLE POSITION = CHANNEL END
                sample_timestamp = primary_channel.end_timestamp

                # Find 5min index for this timestamp
                try:
                    idx_5min = tsla_df.index.searchsorted(sample_timestamp, side='right') - 1
                    if idx_5min < 0:
                        idx_5min = 0
                    if idx_5min >= len(tsla_df):
                        idx_5min = len(tsla_df) - 1
                except Exception as e:
                    results.append({
                        'channel_key': (primary_tf, primary_window, channel_idx),
                        'error': f"Failed to find 5min index: {e}"
                    })
                    continue

                # Check warmup requirement
                if idx_5min < SCANNER_LOOKBACK_5MIN:
                    results.append({'channel_key': (primary_tf, primary_window, channel_idx), 'skipped': True, 'reason': 'warmup'})
                    continue

                # Get data slices for feature extraction
                start_idx = max(0, idx_5min - SCANNER_LOOKBACK_5MIN)
                tsla_slice = tsla_df.iloc[start_idx:idx_5min]
                spy_slice = spy_df.iloc[start_idx:idx_5min]
                vix_slice = vix_df.iloc[start_idx:idx_5min]

                # Extract features at channel end position (with timing)
                feature_start = time.time()
                tf_features = extract_all_tf_features(
                    tsla_df=tsla_slice,
                    spy_df=spy_slice,
                    vix_df=vix_slice,
                    timestamp=sample_timestamp,
                    source_bar_count=idx_5min,
                    include_bar_metadata=True
                )

                tf_features = validate_sample_features(
                    tf_features, sample_timestamp, idx_5min, strict_count=True
                )
                feature_extraction_time_ms = (time.time() - feature_start) * 1000

                # BUILD labels_per_window
                # For PRIMARY channel: use labels directly (no lookup)
                # For OTHER channels: binary search lookup at same timestamp
                labels_per_window = {}
                label_hits = 0
                label_misses = 0

                for w in windows:
                    labels_per_window[w] = {'tsla': {}, 'spy': {}}

                    for t in timeframes:
                        # TSLA labels
                        if t == primary_tf and w == primary_window:
                            # PRIMARY channel - use precomputed labels DIRECTLY
                            tsla_labels = primary_channel.labels
                        else:
                            # Other (tf, window) - lookup channel at this timestamp
                            other_channel = _find_channel_at_timestamp(tsla_slim_map, t, w, sample_timestamp)
                            tsla_labels = other_channel.labels if other_channel else None

                        labels_per_window[w]['tsla'][t] = tsla_labels
                        if tsla_labels is not None and tsla_labels.direction_valid:
                            label_hits += 1
                        else:
                            label_misses += 1

                        # SPY labels - always lookup (we iterate TSLA channels)
                        spy_channel = _find_channel_at_timestamp(spy_slim_map, t, w, sample_timestamp)
                        spy_labels = spy_channel.labels if spy_channel else None
                        labels_per_window[w]['spy'][t] = spy_labels
                        if spy_labels is not None and spy_labels.direction_valid:
                            label_hits += 1
                        else:
                            label_misses += 1

                        # Copy SPY's values into TSLA's spy_* cross-reference fields
                        if spy_labels is not None and tsla_labels is not None:
                            tsla_labels = replace(tsla_labels,
                                spy_source_channel_slope=spy_labels.source_channel_slope,
                                spy_source_channel_intercept=spy_labels.source_channel_intercept,
                                spy_source_channel_std_dev=spy_labels.source_channel_std_dev,
                                spy_source_channel_r_squared=spy_labels.source_channel_r_squared,
                                spy_source_channel_direction=spy_labels.source_channel_direction,
                                spy_source_channel_start_ts=spy_labels.source_channel_start_ts,
                                spy_source_channel_end_ts=spy_labels.source_channel_end_ts,
                                # Copy SPY's next channel labels
                                spy_best_next_channel_direction=spy_labels.best_next_channel_direction,
                                spy_best_next_channel_bars_away=spy_labels.best_next_channel_bars_away,
                                spy_best_next_channel_duration=spy_labels.best_next_channel_duration,
                                spy_best_next_channel_r_squared=spy_labels.best_next_channel_r_squared,
                                spy_best_next_channel_bounce_count=spy_labels.best_next_channel_bounce_count,
                                spy_shortest_next_channel_direction=spy_labels.shortest_next_channel_direction,
                                spy_shortest_next_channel_bars_away=spy_labels.shortest_next_channel_bars_away,
                                spy_shortest_next_channel_duration=spy_labels.shortest_next_channel_duration,
                                spy_small_channels_before_best=spy_labels.small_channels_before_best,
                                # Copy SPY's RSI labels
                                spy_rsi_at_first_break=spy_labels.rsi_at_first_break,
                                spy_rsi_at_permanent_break=spy_labels.rsi_at_permanent_break,
                                spy_rsi_at_channel_end=spy_labels.rsi_at_channel_end,
                                spy_rsi_overbought_at_break=spy_labels.rsi_overbought_at_break,
                                spy_rsi_oversold_at_break=spy_labels.rsi_oversold_at_break,
                                spy_rsi_divergence_at_break=spy_labels.rsi_divergence_at_break,
                                spy_rsi_trend_in_channel=spy_labels.rsi_trend_in_channel,
                                spy_rsi_range_in_channel=spy_labels.rsi_range_in_channel,
                            )
                            # Update the stored TSLA labels with the cross-referenced version
                            labels_per_window[w]['tsla'][t] = tsla_labels

                # Build bar metadata
                bar_metadata = {
                    '5min': {
                        'bar_completion_pct': 1.0,
                        'bars_in_partial': 0,
                        'total_bars': len(tsla_slice),
                        'is_partial': False,
                    }
                }

                # Create sample
                sample = ChannelSample(
                    timestamp=sample_timestamp,
                    channel_end_idx=idx_5min,
                    tf_features=tf_features,
                    labels_per_window=labels_per_window,
                    bar_metadata=bar_metadata,
                    best_window=primary_window  # The window of the PRIMARY channel
                )

                results.append({
                    'channel_key': (primary_tf, primary_window, channel_idx),
                    'sample': sample,
                    'label_hits': label_hits,
                    'label_misses': label_misses,
                    'feature_extraction_time_ms': feature_extraction_time_ms,
                })

            except Exception as e:
                results.append({
                    'channel_key': (primary_tf, primary_window, channel_idx),
                    'error': f"{type(e).__name__}: {str(e)}",
                    'traceback': traceback.format_exc(),
                })

        gc.collect()
        return results

    except Exception as e:
        return [{
            'channel_key': ('error', -1, -1),
            'error': f"BATCH ERROR: {type(e).__name__}: {str(e)}",
            'traceback': traceback.format_exc(),
        }]


class _ProgressFileWriter:
    """File wrapper that reports write progress to tqdm."""
    def __init__(self, file, pbar):
        self.file = file
        self.pbar = pbar

    def write(self, data):
        self.pbar.update(len(data))
        return self.file.write(data)

    def __getattr__(self, attr):
        return getattr(self.file, attr)


def _save_partial_results(samples: List, output_path: str, suffix: str = "_partial"):
    """Save partial results during graceful shutdown."""
    if '.' in output_path:
        base, ext = output_path.rsplit('.', 1)
        partial_path = f"{base}{suffix}.{ext}"
        temp_path = f"{output_path}.tmp"
    else:
        partial_path = f"{output_path}{suffix}"
        temp_path = f"{output_path}.tmp"

    all_samples = []

    if os.path.exists(temp_path):
        try:
            print(f"\n[PARTIAL] Reading samples from temp file...")
            with open(temp_path, 'rb') as f:
                while True:
                    try:
                        sample = pickle.load(f)
                        all_samples.append(sample)
                    except EOFError:
                        break
            print(f"  Loaded {len(all_samples)} samples from temp file")
        except Exception as e:
            print(f"  [WARNING] Failed to read temp file: {e}")

    all_samples.extend(samples)

    if not all_samples:
        print("\n[PARTIAL] No samples to save.")
        return

    try:
        with open(partial_path, 'wb') as f:
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc="Saving partial") as pbar:
                pickle.dump(all_samples, _ProgressFileWriter(f, pbar))
        print(f"\n[SAVED] Partial results ({len(all_samples)} samples) saved to: {partial_path}")

        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"\n[ERROR] Failed to save partial results: {e}")


# Configuration
EXPECTED_FEATURE_COUNT = TOTAL_FEATURES
MIN_LABEL_HIT_RATE = 0.1
MAX_ERRORS_IN_MEMORY = 100
VERBOSE_LOG_INTERVAL = 100  # Log progress every N samples


def _format_time_remaining(seconds: float) -> str:
    """Format seconds into human-readable time string (e.g., '1h 23m 45s')."""
    if seconds < 0:
        return "unknown"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    elif minutes > 0:
        return f"{minutes}m {secs:02d}s"
    else:
        return f"{secs}s"


def _log_sample_progress(
    valid_count: int,
    total_expected: int,
    scan_start: float,
    feature_times_ms: List[float],
    skipped_count: int,
    error_count: int,
    use_tqdm_write: bool = True
) -> None:
    """
    Log verbose progress information for sample generation.

    Args:
        valid_count: Number of valid samples created so far
        total_expected: Total number of samples expected
        scan_start: Start time of the scan (time.time())
        feature_times_ms: List of recent feature extraction times in milliseconds
        skipped_count: Number of skipped samples
        error_count: Number of errors encountered
        use_tqdm_write: If True, use tqdm.write() to avoid overwriting progress bar
    """
    elapsed = time.time() - scan_start
    processed = valid_count + skipped_count + error_count

    # Calculate samples per second and ETA
    if elapsed > 0 and processed > 0:
        samples_per_sec = processed / elapsed
        remaining = total_expected - processed
        eta_seconds = remaining / samples_per_sec if samples_per_sec > 0 else -1
        eta_str = _format_time_remaining(eta_seconds)
    else:
        samples_per_sec = 0.0
        eta_str = "calculating..."

    # Calculate average feature extraction time
    if feature_times_ms:
        avg_feature_time = sum(feature_times_ms) / len(feature_times_ms)
        feature_time_str = f"{avg_feature_time:.1f}ms"
    else:
        feature_time_str = "N/A"

    # Build progress message
    pct = 100 * processed / total_expected if total_expected > 0 else 0
    msg = (
        f"  [PROGRESS] {valid_count}/{total_expected} samples "
        f"({pct:.1f}%) | "
        f"Rate: {samples_per_sec:.1f}/s | "
        f"ETA: {eta_str} | "
        f"Avg feature extraction: {feature_time_str} | "
        f"Skipped: {skipped_count}, Errors: {error_count}"
    )

    if use_tqdm_write:
        tqdm.write(msg)
    else:
        print(msg)


def validate_sample_features(
    tf_features: Dict[str, float],
    timestamp: pd.Timestamp,
    idx: int,
    strict_count: bool = True
) -> Dict[str, float]:
    """Validate extracted features. Raises on any invalid value."""
    n_features = len(tf_features)
    if n_features == 0:
        raise FeatureExtractionError(
            f"No features extracted at idx={idx}, timestamp={timestamp}"
        )

    if strict_count and n_features != EXPECTED_FEATURE_COUNT:
        raise ValidationError(
            f"Feature count mismatch at idx={idx}: got {n_features}, expected {EXPECTED_FEATURE_COUNT}"
        )

    validate_features(tf_features, raise_on_invalid=True)
    return tf_features


def _get_optimal_workers(requested: Optional[int] = None) -> int:
    """Get optimal number of worker processes."""
    if requested is not None:
        return max(1, requested)
    try:
        cpus = cpu_count()
        if cpus is None:
            cpus = 8
        return max(1, cpus - 1)
    except:
        return 4


def scan_channels_two_pass(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    step: int = 10,
    warmup_bars: int = 32760,
    max_samples: Optional[int] = None,
    workers: int = 4,
    batch_size: int = 8,
    progress: bool = True,
    strict: bool = True,
    output_path: Optional[str] = None,
    incremental_path: Optional[str] = None,
    incremental_chunk: int = 1000
) -> List[ChannelSample]:
    """
    Scan for channels using the two-pass labeling system.

    CHANNEL-END SAMPLING ARCHITECTURE:
    1. PASS 1: detect_all_channels() - find all channels
    2. PASS 2: generate_all_labels() - compute labels at channel END
    3. SCAN: Iterate over detected channels - each channel = ONE sample

    Each detected channel produces EXACTLY ONE sample at its end position.
    The --step parameter controls channel detection spacing in Pass 1.

    Args:
        tsla_df: TSLA OHLCV DataFrame with DatetimeIndex
        spy_df: SPY OHLCV DataFrame aligned to TSLA
        vix_df: VIX OHLCV DataFrame aligned to TSLA
        step: Step size for channel detection in Pass 1 (default 10)
        warmup_bars: Minimum 5min bars before first sample (default 32760)
        max_samples: Maximum samples to generate (None = unlimited)
        workers: Number of parallel workers (default 4)
        batch_size: Channels per batch for parallel processing (default 8)
        progress: Show progress bar (default True)
        strict: Raise on errors (default True)
        output_path: Output file path for saving results
        incremental_path: Temp file for incremental writes
        incremental_chunk: Samples to buffer before writing (default 1000)

    Returns:
        List of ChannelSample objects
    """
    global _shutdown_requested
    _shutdown_requested = False

    _setup_multiprocessing()

    original_sigint = signal.signal(signal.SIGINT, _signal_handler)
    original_sigterm = signal.signal(signal.SIGTERM, _signal_handler)

    n_bars = len(tsla_df)

    _check_memory_availability(workers, n_bars)

    # Validate alignment
    if len(spy_df) != n_bars:
        raise DataLoadError(f"TSLA/SPY length mismatch! TSLA has {n_bars} bars, SPY has {len(spy_df)} bars.")
    if len(vix_df) != n_bars:
        raise DataLoadError(f"TSLA/VIX length mismatch! TSLA has {n_bars} bars, VIX has {len(vix_df)} bars.")
    if tsla_df.index[0] != spy_df.index[0] or tsla_df.index[-1] != spy_df.index[-1]:
        raise DataLoadError("TSLA/SPY timestamp mismatch!")
    if tsla_df.index[0] != vix_df.index[0] or tsla_df.index[-1] != vix_df.index[-1]:
        raise DataLoadError("TSLA/VIX timestamp mismatch!")

    print(f"[VALIDATION] Input alignment OK: {n_bars} bars, timestamps aligned")

    print("=" * 60)
    print("V15 Channel Scanner - CHANNEL-END SAMPLING Architecture")
    print("=" * 60)
    print(f"  Workers: {workers}")
    print(f"  Batch size: {batch_size} channels")
    print(f"  Architecture: ONE sample per detected channel at channel END")

    # =========================================================================
    # PASS 1: Pre-compute all channels
    # =========================================================================
    print("\n[PASS 1] Detecting all channels across dataset...")
    print(f"  Timeframes: {list(TIMEFRAMES)}")
    print(f"  Windows: {list(STANDARD_WINDOWS)}")
    print(f"  Channel detection step: {step}")

    pass1_start = time.time()

    print("\n  [PASS 1] Detecting TSLA channels...")
    print(f"           Timeframes: {list(TIMEFRAMES)}")
    print(f"           Windows: {list(STANDARD_WINDOWS)}")
    print(f"           Step: {step}, Workers: {workers}")
    tsla_detect_start = time.time()
    tsla_channel_map, tsla_resampled_dfs = detect_all_channels(
        df=tsla_df,
        timeframes=list(TIMEFRAMES),
        windows=list(STANDARD_WINDOWS),
        step=step,
        min_cycles=1,
        min_gap_bars=5,
        workers=workers,
        verbose=False
    )
    tsla_detect_time = time.time() - tsla_detect_start
    tsla_channels = sum(len(chs) for chs in tsla_channel_map.values())
    print(f"           Completed: {tsla_channels} channels detected in {tsla_detect_time:.1f}s")

    print("\n  [PASS 1] Detecting SPY channels...")
    print(f"           Timeframes: {list(TIMEFRAMES)}")
    print(f"           Windows: {list(STANDARD_WINDOWS)}")
    print(f"           Step: {step}, Workers: {workers}")
    spy_detect_start = time.time()
    spy_channel_map, spy_resampled_dfs = detect_all_channels(
        df=spy_df,
        timeframes=list(TIMEFRAMES),
        windows=list(STANDARD_WINDOWS),
        step=step,
        min_cycles=1,
        min_gap_bars=5,
        workers=workers,
        verbose=False
    )
    spy_detect_time = time.time() - spy_detect_start
    spy_channels = sum(len(chs) for chs in spy_channel_map.values())
    print(f"           Completed: {spy_channels} channels detected in {spy_detect_time:.1f}s")

    pass1_time = time.time() - pass1_start
    print(f"\n  [PASS 1] Summary:")
    print(f"           TSLA: {tsla_channels} channels in {tsla_detect_time:.1f}s")
    print(f"           SPY:  {spy_channels} channels in {spy_detect_time:.1f}s")
    print(f"           Total: {tsla_channels + spy_channels} channels, Pass 1 time: {pass1_time:.1f}s")

    # Save channel maps to disk if output path is specified
    if output_path:
        # Derive base path from output path (remove .pkl extension if present)
        if output_path.endswith('.pkl'):
            output_base = output_path[:-4]
        else:
            output_base = output_path
        channel_map_path = f"{output_base}_channel_map.pkl"

        print(f"\n  [PASS 1] Saving channel maps to {channel_map_path}...")
        channel_map_data = {
            'tsla': tsla_channel_map,
            'spy': spy_channel_map,
        }
        with open(channel_map_path, 'wb') as f:
            pickle.dump(channel_map_data, f)
        print(f"  Saved channel maps ({tsla_channels} TSLA + {spy_channels} SPY channels)")

    # =========================================================================
    # PASS 2: Generate labels at channel END positions
    # =========================================================================
    print("\n[PASS 2] Generating labels from channel maps...")
    pass2_start = time.time()

    # Count total TSLA channels for progress logging
    tsla_total_channels = sum(len(chs) for chs in tsla_channel_map.values())
    print(f"\n  Generating TSLA labels... ({tsla_total_channels} channels to process)")
    tsla_label_start = time.time()
    tsla_labeled_map: LabeledChannelMap = generate_all_labels(
        channel_map=tsla_channel_map,
        resampled_dfs=tsla_resampled_dfs,
        labeling_method="hybrid",
        verbose=True
    )
    tsla_label_time = time.time() - tsla_label_start
    tsla_labeled = sum(len(lcs) for lcs in tsla_labeled_map.values())
    tsla_valid = sum(1 for lcs in tsla_labeled_map.values() for lc in lcs if lc.labels.direction_valid)
    print(f"  TSLA complete: {tsla_labeled} labels generated in {tsla_label_time:.1f}s ({tsla_valid} valid)")

    # Count total SPY channels for progress logging
    spy_total_channels = sum(len(chs) for chs in spy_channel_map.values())
    print(f"\n  Generating SPY labels... ({spy_total_channels} channels to process)")
    spy_label_start = time.time()
    spy_labeled_map: LabeledChannelMap = generate_all_labels(
        channel_map=spy_channel_map,
        resampled_dfs=spy_resampled_dfs,
        labeling_method="hybrid",
        verbose=True
    )
    spy_label_time = time.time() - spy_label_start
    spy_labeled = sum(len(lcs) for lcs in spy_labeled_map.values())
    spy_valid = sum(1 for lcs in spy_labeled_map.values() for lc in lcs if lc.labels.direction_valid)
    print(f"  SPY complete: {spy_labeled} labels generated in {spy_label_time:.1f}s ({spy_valid} valid)")

    pass2_time = time.time() - pass2_start
    print(f"\n  Pass 2 summary: {tsla_labeled + spy_labeled} total labels, {pass2_time:.1f}s total time")

    # Free Pass-1 artifacts
    del tsla_resampled_dfs, spy_resampled_dfs
    del tsla_channel_map, spy_channel_map
    gc.collect()

    # =========================================================================
    # SCAN: Create ONE sample per detected TSLA channel at channel END
    # =========================================================================
    print("\n[SCAN] Creating slim labeled maps...")
    tsla_slim_map = _create_slim_labeled_map(tsla_labeled_map)
    spy_slim_map = _create_slim_labeled_map(spy_labeled_map)

    del tsla_labeled_map, spy_labeled_map
    gc.collect()

    # Build list of (tf, window, channel_idx, end_timestamp) for all valid TSLA channels
    channel_work_items_with_ts = []
    for (tf, window), slim_channels in tsla_slim_map.items():
        for idx, slim_channel in enumerate(slim_channels):
            if slim_channel.channel_valid and slim_channel.labels is not None:
                end_ts = slim_channel.end_timestamp
                try:
                    idx_5min = tsla_df.index.searchsorted(end_ts, side='right') - 1
                    if idx_5min >= warmup_bars:
                        channel_work_items_with_ts.append((tf, window, idx, end_ts))
                except:
                    pass

    # CRITICAL: Sort by end_timestamp to ensure consistent ordering across Python and C++
    # (C++ uses unordered_map which has unpredictable iteration order, so we sort to match)
    channel_work_items_with_ts.sort(key=lambda x: x[3])  # Sort by end_timestamp
    channel_work_items = [(tf, window, idx) for tf, window, idx, _ in channel_work_items_with_ts]

    total_channels_to_process = len(channel_work_items)

    if max_samples is not None and total_channels_to_process > max_samples:
        channel_work_items = channel_work_items[:max_samples]
        print(f"\n[SCAN] Limited to {max_samples} channels (max_samples specified, sorted by timestamp)")

    print(f"\n[SCAN] Starting sample generation...")
    print(f"  Channels to process: {len(channel_work_items)}")
    print(f"  Each channel produces ONE sample at its end position")
    print(f"  Processing mode: {'PARALLEL' if workers > 1 else 'SEQUENTIAL'}")

    samples = []
    errors = []
    error_count = 0
    valid_count = 0
    skipped_count = 0
    label_hit_count = 0
    label_miss_count = 0

    # Verbose logging tracking
    total_samples_expected = len(channel_work_items)
    last_log_count = 0
    feature_extraction_times_ms = []  # Track recent feature extraction times for averaging

    scan_start = time.time()

    try:
        tsla_data = _convert_df_to_pickle_safe(tsla_df)
        spy_data = _convert_df_to_pickle_safe(spy_df)
        vix_data = _convert_df_to_pickle_safe(vix_df)

        batches = [channel_work_items[i:i+batch_size] for i in range(0, len(channel_work_items), batch_size)]
        total_batches = len(batches)
        print(f"  Total batches: {total_batches}")

        if _shutdown_requested:
            print("\n[INTERRUPT] Shutdown was requested during Pass 1/2. Skipping scan phase.")
            return samples

        if workers > 1:
            print(f"\n  Starting parallel processing with {workers} workers...")

            with Pool(
                processes=workers,
                initializer=_init_worker,
                initargs=(tsla_data, spy_data, vix_data, tsla_slim_map, spy_slim_map,
                          list(TIMEFRAMES), list(STANDARD_WINDOWS)),
                maxtasksperchild=50
            ) as pool:
                results_iter = pool.imap_unordered(_process_channel_batch, batches)
                batch_iterator = tqdm(results_iter, total=total_batches, desc="Processing channels", unit="batch") if progress else results_iter

                batches_processed = 0
                for batch_results in batch_iterator:
                    for result in batch_results:
                        if result.get('sample'):
                            samples.append(result['sample'])
                            valid_count += 1
                            label_hit_count += result.get('label_hits', 0)
                            label_miss_count += result.get('label_misses', 0)
                            # Track feature extraction time for averaging
                            if 'feature_extraction_time_ms' in result:
                                feature_extraction_times_ms.append(result['feature_extraction_time_ms'])
                                # Keep only last 100 times for rolling average
                                if len(feature_extraction_times_ms) > 100:
                                    feature_extraction_times_ms.pop(0)
                        elif result.get('error'):
                            error_msg = result['error']
                            if result.get('traceback'):
                                error_msg += f"\nTraceback:\n{result['traceback']}"
                            error_count += 1
                            if len(errors) < MAX_ERRORS_IN_MEMORY:
                                errors.append(error_msg)
                            if strict:
                                pool.terminate()
                                raise ValidationError(error_msg)
                        elif result.get('skipped'):
                            skipped_count += 1

                    batches_processed += 1
                    del batch_results

                    # Verbose logging every VERBOSE_LOG_INTERVAL samples
                    current_processed = valid_count + skipped_count + error_count
                    if current_processed - last_log_count >= VERBOSE_LOG_INTERVAL:
                        _log_sample_progress(
                            valid_count=valid_count,
                            total_expected=total_samples_expected,
                            scan_start=scan_start,
                            feature_times_ms=feature_extraction_times_ms,
                            skipped_count=skipped_count,
                            error_count=error_count,
                            use_tqdm_write=progress
                        )
                        last_log_count = current_processed

                    if incremental_path and len(samples) >= incremental_chunk:
                        flushed = _flush_samples_to_temp(samples, incremental_path)
                        samples.clear()
                        if progress:
                            tqdm.write(f"  [Incremental] Flushed {flushed} samples to disk")

                    if batches_processed % 20 == 0:
                        gc.collect()

                    if _shutdown_requested:
                        print(f"\n[INTERRUPT] Stopping at batch {batches_processed}/{total_batches}")
                        pool.terminate()
                        break

        else:
            print(f"\n  Running in sequential mode...")

            global _WORKER_TSLA_DF, _WORKER_SPY_DF, _WORKER_VIX_DF
            global _WORKER_TSLA_SLIM_MAP, _WORKER_SPY_SLIM_MAP
            global _WORKER_TIMEFRAMES, _WORKER_WINDOWS
            _WORKER_TSLA_DF = _reconstruct_df_from_pickle_safe(tsla_data)
            _WORKER_SPY_DF = _reconstruct_df_from_pickle_safe(spy_data)
            _WORKER_VIX_DF = _reconstruct_df_from_pickle_safe(vix_data)
            _WORKER_TSLA_SLIM_MAP = tsla_slim_map
            _WORKER_SPY_SLIM_MAP = spy_slim_map
            _WORKER_TIMEFRAMES = list(TIMEFRAMES)
            _WORKER_WINDOWS = list(STANDARD_WINDOWS)

            batch_iterator = tqdm(batches, desc="Processing channels", unit="batch") if progress else batches

            for batch in batch_iterator:
                if _shutdown_requested:
                    print(f"\n[INTERRUPT] Stopping scan")
                    break

                batch_results = _process_channel_batch(batch)

                for result in batch_results:
                    if result.get('sample'):
                        samples.append(result['sample'])
                        valid_count += 1
                        label_hit_count += result.get('label_hits', 0)
                        label_miss_count += result.get('label_misses', 0)
                        # Track feature extraction time for averaging
                        if 'feature_extraction_time_ms' in result:
                            feature_extraction_times_ms.append(result['feature_extraction_time_ms'])
                            # Keep only last 100 times for rolling average
                            if len(feature_extraction_times_ms) > 100:
                                feature_extraction_times_ms.pop(0)
                    elif result.get('error'):
                        error_msg = result['error']
                        if result.get('traceback'):
                            error_msg += f"\nTraceback:\n{result['traceback']}"
                        error_count += 1
                        if len(errors) < MAX_ERRORS_IN_MEMORY:
                            errors.append(error_msg)
                        if strict:
                            raise ValidationError(error_msg)
                    elif result.get('skipped'):
                        skipped_count += 1

                # Verbose logging every VERBOSE_LOG_INTERVAL samples
                current_processed = valid_count + skipped_count + error_count
                if current_processed - last_log_count >= VERBOSE_LOG_INTERVAL:
                    _log_sample_progress(
                        valid_count=valid_count,
                        total_expected=total_samples_expected,
                        scan_start=scan_start,
                        feature_times_ms=feature_extraction_times_ms,
                        skipped_count=skipped_count,
                        error_count=error_count,
                        use_tqdm_write=progress
                    )
                    last_log_count = current_processed

                if incremental_path and len(samples) >= incremental_chunk:
                    flushed = _flush_samples_to_temp(samples, incremental_path)
                    samples.clear()
                    if progress:
                        tqdm.write(f"  [Incremental] Flushed {flushed} samples to disk")

    except KeyboardInterrupt:
        print("\n[INTERRUPT] KeyboardInterrupt received. Saving partial results...")
        _shutdown_requested = True

    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)

        if _shutdown_requested and output_path:
            _save_partial_results(samples, output_path)

    scan_time = time.time() - scan_start
    total_time = pass1_time + pass2_time + scan_time

    samples.sort(key=lambda s: s.channel_end_idx)

    # Get memory usage if psutil is available
    memory_info = None
    try:
        import psutil
        process = psutil.Process()
        mem = process.memory_info()
        memory_info = {
            'rss_mb': mem.rss / (1024 * 1024),
            'vms_mb': mem.vms / (1024 * 1024),
            'percent': process.memory_percent(),
        }
    except ImportError:
        pass
    except Exception:
        pass

    # ==========================================================================
    # OVERALL SUMMARY WITH TIMING BREAKDOWN
    # ==========================================================================
    print("\n")
    print("=" * 70)
    if _shutdown_requested:
        print("                         SCAN INTERRUPTED")
    else:
        print("                         SCAN COMPLETE")
    print("=" * 70)

    # --- Results Summary ---
    print("\n" + "-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    print(f"  Total channels processed:     {len(channel_work_items):,}")
    print(f"  Valid samples created:        {valid_count:,}")
    print(f"  Skipped (invalid/no labels):  {skipped_count:,}")
    print(f"  Errors:                       {error_count:,}")

    if valid_count > 0 and samples:
        avg_features = sum(len(s.tf_features) for s in samples) / len(samples)
        print(f"  Average features per sample:  {avg_features:.0f}")

    # Feature extraction timing stats
    if feature_extraction_times_ms:
        avg_feature_time = sum(feature_extraction_times_ms) / len(feature_extraction_times_ms)
        min_feature_time = min(feature_extraction_times_ms)
        max_feature_time = max(feature_extraction_times_ms)
        print(f"\n  Feature Extraction Timing (last {len(feature_extraction_times_ms)} samples):")
        print(f"    Average: {avg_feature_time:.1f}ms")
        print(f"    Min:     {min_feature_time:.1f}ms")
        print(f"    Max:     {max_feature_time:.1f}ms")

    total_lookups = label_hit_count + label_miss_count
    if total_lookups > 0:
        hit_rate = 100 * label_hit_count / total_lookups
        print(f"\n  Label Lookup Stats:")
        print(f"    Hits:   {label_hit_count:,} ({hit_rate:.1f}%)")
        print(f"    Misses: {label_miss_count:,} ({100 - hit_rate:.1f}%)")

    # --- Timing Breakdown ---
    print("\n" + "-" * 70)
    print("TIMING BREAKDOWN")
    print("-" * 70)

    # Calculate percentages
    pass1_pct = (pass1_time / total_time * 100) if total_time > 0 else 0
    pass2_pct = (pass2_time / total_time * 100) if total_time > 0 else 0
    scan_pct = (scan_time / total_time * 100) if total_time > 0 else 0

    print(f"  Pass 1 (channel detection):   {pass1_time:8.1f}s  ({pass1_pct:5.1f}%)")
    print(f"  Pass 2 (label generation):    {pass2_time:8.1f}s  ({pass2_pct:5.1f}%)")
    print(f"  Pass 3 (sample generation):   {scan_time:8.1f}s  ({scan_pct:5.1f}%)")
    print(f"  " + "-" * 40)
    print(f"  TOTAL WALL CLOCK TIME:        {total_time:8.1f}s  (100.0%)")

    # Format total time in human-readable format
    if total_time >= 3600:
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        print(f"                                ({hours}h {minutes}m {seconds:.1f}s)")
    elif total_time >= 60:
        minutes = int(total_time // 60)
        seconds = total_time % 60
        print(f"                                ({minutes}m {seconds:.1f}s)")

    # --- Performance Metrics ---
    print("\n" + "-" * 70)
    print("PERFORMANCE METRICS")
    print("-" * 70)

    # Samples per second
    if total_time > 0 and valid_count > 0:
        samples_per_sec = valid_count / total_time
        samples_per_sec_scan = valid_count / scan_time if scan_time > 0 else 0
        print(f"  Overall throughput:           {samples_per_sec:.2f} samples/sec")
        print(f"  Pass 3 throughput:            {samples_per_sec_scan:.2f} samples/sec")

    # Channels per second in Pass 1
    if pass1_time > 0:
        channels_per_sec_p1 = (tsla_channels + spy_channels) / pass1_time
        print(f"  Pass 1 channel detection:     {channels_per_sec_p1:.2f} channels/sec")

    # Labels per second in Pass 2
    if pass2_time > 0:
        labels_per_sec_p2 = (tsla_labeled + spy_labeled) / pass2_time
        print(f"  Pass 2 label generation:      {labels_per_sec_p2:.2f} labels/sec")

    # --- Memory Usage ---
    if memory_info:
        print("\n" + "-" * 70)
        print("MEMORY USAGE")
        print("-" * 70)
        print(f"  Resident Set Size (RSS):      {memory_info['rss_mb']:.1f} MB")
        print(f"  Virtual Memory Size (VMS):    {memory_info['vms_mb']:.1f} MB")
        print(f"  Memory Percent:               {memory_info['percent']:.1f}%")

    print("\n" + "=" * 70)
    print(f"  COMPLETE: {valid_count:,} samples generated in {total_time:.1f}s")
    print("=" * 70)

    if errors:
        print(f"\n{'=' * 60}")
        print(f"ERRORS ENCOUNTERED: {len(errors)}")
        print(f"{'=' * 60}")
        for i, err in enumerate(errors[:10]):
            print(f"\n[ERROR {i+1}]")
            for line in str(err).split('\n')[:5]:
                print(f"  {line}")
        if len(errors) > 10:
            print(f"\n  ... and {len(errors) - 10} more errors")

    # Final flush for incremental mode
    if incremental_path:
        if samples:
            flushed = _flush_samples_to_temp(samples, incremental_path)
            print(f"\n  [Incremental] Final flush: {flushed} samples")
            samples.clear()

        if output_path:
            sample_count = _consolidate_temp_to_final(incremental_path, output_path, progress=progress)
            print(f"  [Incremental] Wrote {sample_count} samples directly to {output_path}")
            return []
        else:
            print(f"  [Incremental] Reading samples back into memory...")
            with open(incremental_path, 'rb') as f:
                while True:
                    try:
                        sample = pickle.load(f)
                        samples.append(sample)
                    except EOFError:
                        break
            os.remove(incremental_path)
            samples.sort(key=lambda s: s.channel_end_idx)

    return samples


def _flush_samples_to_temp(samples: list, temp_file_path: str) -> int:
    """Write samples to temp file and return count written."""
    if not samples or not temp_file_path:
        return 0

    count = len(samples)
    with open(temp_file_path, 'ab') as f:
        for sample in samples:
            pickle.dump(sample, f)

    return count


def _consolidate_temp_to_final(temp_file_path: str, final_path: str, progress: bool = True) -> int:
    """Read all samples from temp file and write to final pickle."""
    if not os.path.exists(temp_file_path):
        print(f"\n  [Incremental] No samples generated")
        with open(final_path, 'wb') as f:
            pickle.dump([], f)
        return 0

    samples = []

    print(f"\n  [Incremental] Consolidating temp file to final output...")
    with open(temp_file_path, 'rb') as f:
        while True:
            try:
                sample = pickle.load(f)
                samples.append(sample)
            except EOFError:
                break

    samples.sort(key=lambda s: s.channel_end_idx)
    print(f"  [Incremental] Read and sorted {len(samples)} samples from temp file")

    print(f"  [Incremental] Writing to {final_path}...")
    count = len(samples)
    with open(final_path, 'wb') as f:
        if progress:
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc="Saving") as pbar:
                pickle.dump(samples, _ProgressFileWriter(f, pbar))
        else:
            pickle.dump(samples, f)

    os.remove(temp_file_path)
    print(f"  [Incremental] Cleaned up temp file")

    del samples
    return count


def main():
    """Main entry point for the scanner."""
    _setup_multiprocessing()

    parser = argparse.ArgumentParser(description='V15 Two-Pass Channel Scanner (Channel-End Architecture)')
    parser.add_argument('--step', type=int, default=10,
                        help='Step size for CHANNEL DETECTION in Pass 1 (default: 10)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to generate')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path for samples (pickle format)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: auto-detect)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Channels per batch for parallel processing (default: 8)')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel processing')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory path (default: data)')
    parser.add_argument('--incremental', action='store_true',
                        help='Write results incrementally to disk')
    parser.add_argument('--incremental-chunk', type=int, default=1000,
                        help='Number of samples to buffer before writing (default: 1000)')
    args = parser.parse_args()

    if args.incremental_chunk != 1000:
        args.incremental = True

    if args.incremental and not args.output:
        print("\n" + "=" * 60)
        print("WARNING: --incremental has no effect without --output")
        print("=" * 60 + "\n")
        args.incremental = False

    if args.no_parallel:
        workers = 1
        batch_size = 1
    else:
        workers = _get_optimal_workers(args.workers)
        batch_size = args.batch_size

    print("=" * 60)
    print("V15 Channel Scanner - CHANNEL-END SAMPLING Architecture")
    print("=" * 60)
    print(f"\nArchitecture: ONE sample per detected channel at channel END")
    print(f"  - Each channel produces exactly one sample")
    print(f"  - Sample position = channel end position")
    print(f"  - --step controls channel detection step, not sample step")

    print(f"\nConfiguration:")
    print(f"  Channel detection step: {args.step}")
    print(f"  Max samples: {args.max_samples if args.max_samples else 'unlimited'}")
    print(f"  Output file: {args.output if args.output else 'none'}")
    print(f"  Workers: {workers}")

    temp_file_path = None
    if args.incremental and args.output:
        temp_file_path = args.output + '.tmp'
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    print(f"\nLoading market data from {args.data_dir}...")
    tsla, spy, vix = load_market_data(args.data_dir)
    print(f"Loaded {len(tsla)} bars")
    print(f"Date range: {tsla.index[0]} to {tsla.index[-1]}")

    _check_memory_availability(workers, len(tsla))

    print(f"\nExpected feature count: {EXPECTED_FEATURE_COUNT}")

    print(f"\nRunning TWO-PASS scan (channel detection step={args.step})...")
    samples = scan_channels_two_pass(
        tsla, spy, vix,
        step=args.step,
        max_samples=args.max_samples,
        workers=workers,
        batch_size=batch_size,
        progress=True,
        strict=True,
        output_path=args.output,
        incremental_path=temp_file_path if args.incremental else None,
        incremental_chunk=args.incremental_chunk
    )

    # Handle output
    if not samples and args.incremental and args.output:
        print(f"\nSamples written directly to {args.output} (incremental mode)")
        with open(args.output, 'rb') as f:
            all_samples = pickle.load(f)
        print(f"  Total samples: {len(all_samples)}")
        if all_samples:
            print(f"\nFirst sample: {all_samples[0].timestamp}")
            print(f"Last sample: {all_samples[-1].timestamp}")
        del all_samples
    else:
        print(f"\nGenerated {len(samples)} samples")

        if samples:
            sample = samples[0]
            print(f"\nFirst sample details:")
            print(f"  Timestamp: {sample.timestamp}")
            print(f"  Best window: {sample.best_window}")
            print(f"  Feature count: {len(sample.tf_features)}")

            feature_names = list(sample.tf_features.keys())
            print(f"\n  Sample feature names (first 10):")
            for name in feature_names[:10]:
                print(f"    - {name}: {sample.tf_features[name]:.4f}")

            print(f"\nLast sample: {samples[-1].timestamp}")

        if args.output and samples:
            print(f"\nSaving {len(samples)} samples to {args.output}...")
            with open(args.output, 'wb') as f:
                with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc="Saving") as pbar:
                    pickle.dump(samples, _ProgressFileWriter(f, pbar))
            print(f"Saved successfully!")


# Alias for backward compatibility
scan_channels = scan_channels_two_pass


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
