"""
PyTorch Dataset for Channel Prediction Training

This module provides:
1. ChannelDataset - Main PyTorch Dataset for loading channel samples
2. Data preparation utilities for scanning and caching valid channels
3. Batch collation functions for DataLoader
4. Train/val/test splitting by time periods

The dataset yields (features_dict, labels_dict) tuples ready for model training.
"""

import numpy as np
import pandas as pd
import torch
import warnings
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import json
from datetime import datetime

from ..core.channel import detect_channel, Channel
from ..core.timeframe import resample_ohlc, TIMEFRAMES, BARS_PER_TF
from ..features.full_features import extract_full_features, features_to_tensor_dict, FullFeatures
from .labels import generate_labels, generate_labels_per_tf, ChannelLabels, labels_to_dict, labels_to_array


# =============================================================================
# Cache Version Management
# =============================================================================

# Increment this version when cache format changes:
# - Changes to feature extraction logic
# - Changes to label generation
# - Changes to ChannelSample structure
# - Changes to warmup period or timeframe handling
#
# Version History:
# - v7.0.0: Initial v7 cache format
# - v7.1.0: Added lookforward_bars parameter
# - v7.2.0: Fixed label generation edge cases
# - v7.3.0: Increased to 32,760-bar warmup (420 days) for monthly window=20
# - v8.0.0: Native per-TF labels - labels generated independently for each timeframe
#           using generate_labels_per_tf(). Each TF has its own duration_bars in native
#           TF resolution (not scaled from 5min). labels is now Dict[str, ChannelLabels].
# - v9.0.0: Added trigger_tf as prediction target with separate validity masks.
#           ChannelLabels now has: break_trigger_tf as int (0-20), and per-label validity
#           flags (duration_valid, direction_valid, trigger_tf_valid, new_channel_valid).
#           Dataset returns separate masks for each label type.
CACHE_VERSION = "v9.0.0"

# Supported cache versions for migration (read-only compatibility)
# These older versions can be loaded but will trigger rebuild recommendation
COMPATIBLE_CACHE_VERSIONS = ["v8.0.0", "v7.3.0", "v7.2.0", "v7.1.0"]


def get_cache_metadata_path(cache_path: Path) -> Path:
    """Get the metadata path for a cache file."""
    return cache_path.with_suffix('.json')


def get_cache_metadata(cache_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load cache metadata from JSON file.

    Args:
        cache_path: Path to cache file (.pkl)

    Returns:
        Metadata dict or None if not found/invalid
    """
    meta_path = get_cache_metadata_path(cache_path)
    if not meta_path.exists():
        return None

    try:
        with open(meta_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def is_cache_valid(cache_path: Path, strict: bool = True) -> bool:
    """
    Check if cached data exists and has valid version.

    NOTE: This only checks version. Use validate_cache_params() for full validation.

    Args:
        cache_path: Path to cache file
        strict: If True, only exact version match is valid.
                If False, also accept COMPATIBLE_CACHE_VERSIONS (with warning).

    Returns:
        True if cache exists and version matches, False otherwise
    """
    if not cache_path.exists():
        return False

    meta_path = get_cache_metadata_path(cache_path)
    if not meta_path.exists():
        print(f"Cache metadata missing at {meta_path}")
        return False

    try:
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        cached_version = metadata.get('cache_version', 'unknown')

        # Exact match - always valid
        if cached_version == CACHE_VERSION:
            return True

        # Compatible version check (only if not strict)
        if not strict and cached_version in COMPATIBLE_CACHE_VERSIONS:
            print(f"Cache version {cached_version} is compatible but outdated.")
            print(f"  Current version: {CACHE_VERSION}")
            print(f"  Recommend rebuilding cache for native per-TF labels.")
            return True

        # Version mismatch
        print(f"Cache version mismatch: found {cached_version}, expected {CACHE_VERSION}")
        if cached_version in COMPATIBLE_CACHE_VERSIONS:
            print(f"  (This version is loadable with strict=False, but rebuild recommended)")
        return False

    except Exception as e:
        print(f"Error reading cache metadata: {e}")
        return False


def validate_cache_params(
    cache_path: Path,
    window: int,
    step: int,
    min_cycles: int = 1,
    max_scan: int = 500,
    return_threshold: int = 20,
    include_history: bool = False,
    lookforward_bars: int = 200,
    native_per_tf_labels: bool = True  # v8.0.0: Native per-TF label generation
) -> Tuple[bool, List[str]]:
    """
    Validate that cache parameters match requested parameters.

    Args:
        cache_path: Path to cache file
        window, step, etc: Requested parameters to validate against cache
        native_per_tf_labels: v8.0.0 feature - whether labels are generated per-TF

    Returns:
        Tuple of (is_valid, list_of_mismatches)
        - is_valid: True if all params match (or cache doesn't exist)
        - list_of_mismatches: Human-readable list of parameter mismatches
    """
    metadata = get_cache_metadata(cache_path)
    if metadata is None:
        return True, []  # No cache = no mismatch

    mismatches = []

    # Check each parameter
    param_checks = [
        ('window', window, metadata.get('window')),
        ('step', step, metadata.get('step')),
        ('min_cycles', min_cycles, metadata.get('min_cycles')),
        ('max_scan', max_scan, metadata.get('max_scan')),
        ('return_threshold', return_threshold, metadata.get('return_threshold')),
        ('include_history', include_history, metadata.get('include_history')),
        ('lookforward_bars', lookforward_bars, metadata.get('lookforward_bars')),
    ]

    for param_name, requested, cached in param_checks:
        if cached is not None and cached != requested:
            mismatches.append(f"{param_name}: cached={cached}, requested={requested}")

    # v8.0.0: Check native per-TF labels flag
    cached_native_labels = metadata.get('native_per_tf_labels', False)
    if native_per_tf_labels and not cached_native_labels:
        mismatches.append(
            f"native_per_tf_labels: cached={cached_native_labels}, requested={native_per_tf_labels}"
        )

    return len(mismatches) == 0, mismatches


def get_cache_summary(cache_path: Path) -> Optional[Dict[str, Any]]:
    """
    Get a summary of cached data for display.

    Args:
        cache_path: Path to cache file

    Returns:
        Dict with cache summary info, or None if cache doesn't exist
    """
    if not cache_path.exists():
        return None

    metadata = get_cache_metadata(cache_path)
    if metadata is None:
        return None

    # Get file size
    try:
        file_size_mb = cache_path.stat().st_size / (1024 * 1024)
    except Exception:
        file_size_mb = 0

    cached_version = metadata.get('cache_version', 'unknown')

    return {
        'exists': True,
        'path': str(cache_path),
        'file_size_mb': round(file_size_mb, 1),
        'cache_version': cached_version,
        'version_valid': cached_version == CACHE_VERSION,
        'version_compatible': cached_version in COMPATIBLE_CACHE_VERSIONS,
        'needs_migration': cached_version in COMPATIBLE_CACHE_VERSIONS,
        'num_samples': metadata.get('num_samples', 0),
        'window': metadata.get('window'),
        'step': metadata.get('step'),
        'min_cycles': metadata.get('min_cycles'),
        'max_scan': metadata.get('max_scan'),
        'return_threshold': metadata.get('return_threshold'),
        'include_history': metadata.get('include_history'),
        'lookforward_bars': metadata.get('lookforward_bars'),
        'start_date': metadata.get('start_date'),
        'end_date': metadata.get('end_date'),
        'created_at': metadata.get('created_at'),
        # v8.0.0 fields
        'native_per_tf_labels': metadata.get('native_per_tf_labels', False),
        'valid_tf_labels': metadata.get('valid_tf_labels', []),
        'label_generation_mode': metadata.get('label_generation_mode', 'legacy'),
    }


# =============================================================================
# Date Range Validation Helpers
# =============================================================================

def validate_date_range(
    date_str: str,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Validate if a date string falls within an optional min/max range.

    Args:
        date_str: Date string to validate
        min_date: Optional minimum date
        max_date: Optional maximum date

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        timestamp = pd.Timestamp(date_str)
    except Exception as e:
        return False, f"Invalid date format '{date_str}': {str(e)}"

    if min_date is not None:
        try:
            min_timestamp = pd.Timestamp(min_date)
            if timestamp < min_timestamp:
                return False, f"Date {date_str} is before minimum {min_date}"
        except Exception as e:
            return False, f"Invalid min_date format '{min_date}': {str(e)}"

    if max_date is not None:
        try:
            max_timestamp = pd.Timestamp(max_date)
            if timestamp > max_timestamp:
                return False, f"Date {date_str} exceeds maximum {max_date}"
        except Exception as e:
            return False, f"Invalid max_date format '{max_date}': {str(e)}"

    return True, ""


def get_data_date_range(
    data_dir: Path,
    file_prefix: str = "TSLA"
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Quickly load first and last timestamp from a CSV file.

    Args:
        data_dir: Directory containing CSV files
        file_prefix: File prefix - "TSLA", "SPY", or "VIX"

    Returns:
        Tuple of (start_date, end_date) as pd.Timestamp objects
    """
    if file_prefix in ["TSLA", "SPY"]:
        file_path = data_dir / f"{file_prefix}_1min.csv"
        date_col = "timestamp"
    elif file_prefix == "VIX":
        file_path = data_dir / "VIX_History.csv"
        date_col = "DATE"
    else:
        raise ValueError(f"Unknown file_prefix: {file_prefix}")

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    # Read first row
    first_chunk = pd.read_csv(
        file_path,
        usecols=[date_col],
        nrows=1,
        parse_dates=[date_col]
    )
    start_date = first_chunk[date_col].iloc[0]

    # Read last rows efficiently
    last_date = None
    for chunk in pd.read_csv(
        file_path,
        usecols=[date_col],
        chunksize=100000,
        parse_dates=[date_col]
    ):
        if not chunk.empty:
            last_date = chunk[date_col].iloc[-1]

    return start_date, last_date


@dataclass
class ChannelSample:
    """
    A single training sample consisting of a valid channel + its features and labels.

    Attributes:
        timestamp: When this channel ends (last bar of detection window)
        channel_end_idx: Index in the full dataset where channel ends
        channel: The detected Channel object
        features: FullFeatures extracted at channel end
        labels: Dict mapping TF names to ChannelLabels (native per-TF labels).
                Each TF has independently generated labels with proper scaling.
                Some TFs may have None if channel detection failed at that TF.

    Note:
        Native per-TF labels are generated by:
        1. Resampling OHLC to each timeframe
        2. Detecting channel at that TF's native resolution
        3. Scanning forward at that TF's bar count
        This means duration_bars is in native TF bars (not 5min bars scaled).
    """
    timestamp: pd.Timestamp
    channel_end_idx: int
    channel: Channel
    features: FullFeatures
    labels: Dict[str, ChannelLabels]  # Keyed by TF name; None values for failed TFs


class ChannelDataset(Dataset):
    """
    PyTorch Dataset for channel prediction.

    Loads pre-cached channel samples or generates them on-the-fly.
    Each sample contains:
    - features_dict: Dict of feature tensors from full_features.py
    - labels_dict: Dict of label values from labels.py

    Args:
        samples: List of ChannelSample objects
        transform: Optional transform to apply to features
        augment: Whether to apply data augmentation
    """

    def __init__(
        self,
        samples: List[ChannelSample],
        transform: Optional[callable] = None,
        augment: bool = False,
        augment_noise_std: float = 0.01,
        augment_time_shift: int = 0
    ):
        self.samples = samples
        self.transform = transform
        self.augment = augment
        self.augment_noise_std = augment_noise_std
        self.augment_time_shift = augment_time_shift

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get a single sample.

        Returns:
            Tuple of (features_dict, labels_dict)
            - features_dict: Dict mapping feature names to tensors
            - labels_dict: Dict with label values (duration, break_dir, etc.)

        v9.0.0 labels_dict structure:
            Per-TF labels (shape [11]):
                - duration: float, native TF bars
                - direction: long, 0=DOWN, 1=UP
                - next_channel: long, 0=BEAR, 1=SIDEWAYS, 2=BULL
                - trigger_tf: long, 0-20 (NO_TRIGGER or TF+boundary)

            Per-label validity masks (shape [11]):
                - duration_valid: float, 1.0 if duration is valid
                - direction_valid: float, 1.0 if direction is valid
                - next_channel_valid: float, 1.0 if next_channel is valid
                - trigger_tf_valid: float, 1.0 if trigger_tf is valid

            Aggregate labels (scalars):
                - duration_bars: float, 5min reference
                - permanent_break: long, 0 or 1
        """
        sample = self.samples[idx]

        # Convert features to tensor dict
        features_dict = features_to_tensor_dict(sample.features)

        # Convert to PyTorch tensors
        features_tensors = {
            k: torch.from_numpy(v).float()
            for k, v in features_dict.items()
        }

        # Apply augmentation if enabled
        if self.augment:
            features_tensors = self._augment_features(features_tensors)

        # Convert per-TF labels to tensors
        # sample.labels is Dict[str, ChannelLabels] keyed by TF name
        # Extract native per-TF labels with separate validity masks
        duration_list = []
        direction_list = []
        next_channel_list = []
        trigger_tf_list = []

        # Separate validity masks for each label type
        duration_valid_list = []
        direction_valid_list = []
        next_channel_valid_list = []
        trigger_tf_valid_list = []

        # Default values for missing TF labels
        default_duration = 0.0
        default_direction = 0  # DOWN
        default_next_channel = 1  # SIDEWAYS
        default_trigger_tf = 0  # NO_TRIGGER

        for tf in TIMEFRAMES:
            tf_labels = sample.labels.get(tf)
            if tf_labels is not None:
                # Native per-TF labels - duration is already in the TF's native bars
                duration_list.append(float(tf_labels.duration_bars))
                direction_list.append(int(tf_labels.break_direction))
                next_channel_list.append(int(tf_labels.new_channel_direction))

                # trigger_tf is int (0-20) in v9.0.0 format
                trigger_tf_val = tf_labels.break_trigger_tf
                trigger_tf_list.append(int(trigger_tf_val) if trigger_tf_val is not None else 0)

                # Per-label validity flags (v9.0.0 format required)
                duration_valid_list.append(1.0 if tf_labels.duration_valid else 0.0)
                direction_valid_list.append(1.0 if tf_labels.direction_valid else 0.0)
                next_channel_valid_list.append(1.0 if tf_labels.new_channel_valid else 0.0)
                trigger_tf_valid_list.append(1.0 if tf_labels.trigger_tf_valid else 0.0)
            else:
                # Missing TF label - use defaults and mark all as invalid
                duration_list.append(default_duration)
                direction_list.append(default_direction)
                next_channel_list.append(default_next_channel)
                trigger_tf_list.append(default_trigger_tf)
                duration_valid_list.append(0.0)
                direction_valid_list.append(0.0)
                next_channel_valid_list.append(0.0)
                trigger_tf_valid_list.append(0.0)

        labels_dict = {
            # Per-timeframe labels (native, not scaled) - [11]
            'duration': torch.tensor(duration_list, dtype=torch.float32),
            'direction': torch.tensor(direction_list, dtype=torch.long),
            'next_channel': torch.tensor(next_channel_list, dtype=torch.long),
            'trigger_tf': torch.tensor(trigger_tf_list, dtype=torch.long),

            # Per-label validity masks - [11]
            'duration_valid': torch.tensor(duration_valid_list, dtype=torch.float32),
            'direction_valid': torch.tensor(direction_valid_list, dtype=torch.float32),
            'next_channel_valid': torch.tensor(next_channel_valid_list, dtype=torch.float32),
            'trigger_tf_valid': torch.tensor(trigger_tf_valid_list, dtype=torch.float32),

            # Keep 5min originals for reference/logging (use 5min if available, else first valid)
            'duration_bars': torch.tensor(
                sample.labels.get('5min').duration_bars if sample.labels.get('5min') else duration_list[0],
                dtype=torch.float32
            ),
            'permanent_break': torch.tensor(
                int(sample.labels.get('5min').permanent_break) if sample.labels.get('5min') else 0,
                dtype=torch.long
            ),
        }

        # Apply transform if provided
        if self.transform:
            features_tensors = self.transform(features_tensors)

        return features_tensors, labels_dict

    def _augment_features(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply data augmentation to features.

        Augmentation strategies:
        1. Add small Gaussian noise to continuous features
        2. Time shift (not implemented yet - requires regenerating features)

        Args:
            features: Dict of feature tensors

        Returns:
            Augmented features dict
        """
        augmented = {}

        for key, tensor in features.items():
            # Add noise to all continuous features except binary flags
            if self.augment_noise_std > 0:
                noise = torch.randn_like(tensor) * self.augment_noise_std
                augmented[key] = tensor + noise
            else:
                augmented[key] = tensor

        return augmented


def load_market_data(
    data_dir: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load TSLA, SPY, and VIX data with proper date alignment.

    Ensures all dataframes align to TSLA's index via reindexing.

    Args:
        data_dir: Directory containing CSV files
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        verbose: Print alignment information

    Returns:
        Tuple of (tsla_df, spy_df, vix_df) - all with aligned indices
    """
    # Load TSLA 1min data
    tsla_path = data_dir / "TSLA_1min.csv"
    tsla_df = pd.read_csv(tsla_path, parse_dates=['timestamp'])
    tsla_df.set_index('timestamp', inplace=True)
    tsla_df.columns = [c.lower() for c in tsla_df.columns]
    tsla_df = resample_ohlc(tsla_df, '5min')

    # Load SPY 1min data
    spy_path = data_dir / "SPY_1min.csv"
    spy_df = pd.read_csv(spy_path, parse_dates=['timestamp'])
    spy_df.set_index('timestamp', inplace=True)
    spy_df.columns = [c.lower() for c in spy_df.columns]
    spy_df = resample_ohlc(spy_df, '5min')

    # Load VIX daily data
    vix_path = data_dir / "VIX_History.csv"
    vix_df = pd.read_csv(vix_path, parse_dates=['DATE'])
    vix_df.set_index('DATE', inplace=True)
    vix_df.columns = [c.lower() for c in vix_df.columns]

    if verbose:
        print("Raw data loaded:")
        print(f"  TSLA: {len(tsla_df)} bars ({tsla_df.index[0]} to {tsla_df.index[-1]})")
        print(f"  SPY:  {len(spy_df)} bars ({spy_df.index[0]} to {spy_df.index[-1]})")
        print(f"  VIX:  {len(vix_df)} bars ({vix_df.index[0]} to {vix_df.index[-1]})")

    # Apply user filters
    if start_date:
        tsla_df = tsla_df[tsla_df.index >= start_date]
        spy_df = spy_df[spy_df.index >= start_date]
        vix_df = vix_df[vix_df.index >= start_date]

    if end_date:
        tsla_df = tsla_df[tsla_df.index <= end_date]
        spy_df = spy_df[spy_df.index <= end_date]
        vix_df = vix_df[vix_df.index <= end_date]

    # Find intersection of date ranges
    tsla_dates = set(tsla_df.index.date)
    spy_dates = set(spy_df.index.date)
    vix_dates = set(vix_df.index.date)

    common_dates = tsla_dates & spy_dates & vix_dates

    if not common_dates:
        raise ValueError("No overlapping dates between TSLA, SPY, and VIX")

    intersection_start = min(common_dates)
    intersection_end = max(common_dates)

    # Filter to intersection
    tsla_df = tsla_df[(tsla_df.index.date >= intersection_start) & (tsla_df.index.date <= intersection_end)]
    spy_df = spy_df[(spy_df.index.date >= intersection_start) & (spy_df.index.date <= intersection_end)]
    vix_df = vix_df[(vix_df.index.date >= intersection_start) & (vix_df.index.date <= intersection_end)]

    # Reindex SPY and VIX to TSLA's index
    spy_aligned = spy_df.reindex(tsla_df.index, method='ffill')
    vix_aligned = vix_df.reindex(tsla_df.index, method='ffill')

    # Drop NaNs
    combined = pd.concat([tsla_df, spy_aligned, vix_aligned], axis=1)
    valid_mask = ~combined.isna().any(axis=1)

    tsla_aligned = tsla_df[valid_mask].copy()
    spy_aligned = spy_aligned[valid_mask].copy()
    vix_aligned = vix_aligned[valid_mask].copy()

    if verbose:
        print(f"\nAligned data:")
        print(f"  All series: {len(tsla_aligned)} bars ({tsla_aligned.index[0]} to {tsla_aligned.index[-1]})")

    return tsla_aligned, spy_aligned, vix_aligned


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
    progress: bool = True
) -> Tuple[List[ChannelSample], int]:
    """
    Scan through historical data to find all valid channels and generate samples.

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

    Returns:
        List of ChannelSample objects
    """
    samples = []

    # Statistics for summary (instead of per-position spam)
    stats = {
        'total_scanned': 0,
        'invalid_channel': 0,
        'feature_failed': 0,
        'label_failed': 0,
        'no_valid_labels': 0,
        'valid_samples': 0,
    }

    # Warmup period: ensure adequate data for all timeframes
    # Need ~32,760 bars (420 trading days) for monthly channels to have window=20 native monthly bars
    # This ensures monthly timeframes have proper statistical validity (20 bars for regression)
    # 3-month will still be weak (~6.7 bars) but acceptable with quality scoring
    min_warmup_bars = max(window, 32760)  # At least 32,760 bars (20 months) or window, whichever larger

    # Need enough data for warmup + forward scan
    min_required = min_warmup_bars + max_scan

    # Align SPY and VIX with TSLA timestamps
    # For SPY (5min), reindex to match TSLA timestamps and forward-fill gaps
    spy_aligned = spy_df.reindex(tsla_df.index, method='ffill')

    # For VIX (daily), forward-fill to match 5min timestamps
    vix_aligned = vix_df.reindex(tsla_df.index, method='ffill')

    # Scan through data with sliding window
    start_idx = min_warmup_bars
    end_idx = len(tsla_df) - max_scan

    indices = range(start_idx, end_idx, step)
    if progress:
        indices = tqdm(indices, desc="Scanning channels")

    for i in indices:
        stats['total_scanned'] += 1

        # Get data window
        tsla_window = tsla_df.iloc[:i]
        spy_window = spy_aligned.iloc[:i]
        vix_window = vix_aligned.iloc[:i]

        if len(tsla_window) < window or len(spy_window) < window:
            continue

        # Detect channel on TSLA
        channel = detect_channel(tsla_window, window=window, min_cycles=min_cycles)

        if not channel.valid:
            stats['invalid_channel'] += 1
            continue

        # Extract features at this point
        try:
            features = extract_full_features(
                tsla_window,
                spy_window,
                vix_window,
                window=window,
                include_history=include_history,
                lookforward_bars=lookforward_bars
            )
        except Exception as e:
            # Skip if feature extraction fails (only log first occurrence)
            stats['feature_failed'] += 1
            if stats['feature_failed'] == 1 and progress:
                tqdm.write(f"Feature extraction failed (first error): {e}")
            continue

        # Generate native per-TF labels by scanning forward at each timeframe
        try:
            labels_per_tf = generate_labels_per_tf(
                df=tsla_df.iloc[:i + max_scan],  # Include forward data for label generation
                window=window,
                max_scan=max_scan,
                return_threshold=return_threshold
            )
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

        # Create sample with per-TF labels dict
        sample = ChannelSample(
            timestamp=tsla_df.index[i - 1],
            channel_end_idx=i - 1,
            channel=channel,
            features=features,
            labels=labels_per_tf
        )

        samples.append(sample)
        stats['valid_samples'] += 1

    # Print summary statistics instead of per-position spam
    if progress and stats['total_scanned'] > 0:
        print(f"\nChannel scanning summary:")
        print(f"  Total positions scanned: {stats['total_scanned']}")
        print(f"  Valid samples created:   {stats['valid_samples']} ({100*stats['valid_samples']/stats['total_scanned']:.1f}%)")
        if stats['invalid_channel'] > 0:
            print(f"  Invalid channels:        {stats['invalid_channel']} ({100*stats['invalid_channel']/stats['total_scanned']:.1f}%)")
        if stats['feature_failed'] > 0:
            print(f"  Feature extraction failed: {stats['feature_failed']}")
        if stats['label_failed'] > 0:
            print(f"  Label generation failed: {stats['label_failed']}")
        if stats['no_valid_labels'] > 0:
            print(f"  No valid TF labels:      {stats['no_valid_labels']} ({100*stats['no_valid_labels']/stats['total_scanned']:.1f}%)")

    return samples, min_warmup_bars


def cache_samples(
    samples: List[ChannelSample],
    cache_path: Path,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Cache samples to disk for faster loading.

    Args:
        samples: List of ChannelSample objects
        cache_path: Path to save cache file (.pkl)
        metadata: Optional metadata to save with cache
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Save samples
    with open(cache_path, 'wb') as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save metadata as JSON with cache version
    meta_path = get_cache_metadata_path(cache_path)
    meta_serializable = {'cache_version': CACHE_VERSION}

    if metadata:
        # Convert datetime objects to strings for JSON
        for k, v in metadata.items():
            if isinstance(v, (datetime, pd.Timestamp)):
                meta_serializable[k] = str(v)
            else:
                meta_serializable[k] = v

    with open(meta_path, 'w') as f:
        json.dump(meta_serializable, f, indent=2)

    print(f"Cached {len(samples)} samples to {cache_path}")
    print(f"Cache version: {CACHE_VERSION}")


def load_cached_samples(
    cache_path: Path,
    migrate_labels: bool = False
) -> Tuple[List[ChannelSample], Dict[str, Any]]:
    """
    Load cached samples from disk with version-aware handling.

    For v7.x caches loaded into v8.0.0, can optionally migrate labels format.
    Migration converts single ChannelLabels to Dict[str, ChannelLabels] format.

    Args:
        cache_path: Path to cache file (.pkl)
        migrate_labels: If True and loading v7.x cache, convert labels to per-TF dict

    Returns:
        Tuple of (samples, load_info)
        - samples: List of ChannelSample objects
        - load_info: Dict with version info and any migration warnings
    """
    # Get version info before loading
    metadata = get_cache_metadata(cache_path)
    cached_version = metadata.get('cache_version', 'unknown') if metadata else 'unknown'
    native_labels = metadata.get('native_per_tf_labels', False) if metadata else False

    with open(cache_path, 'rb') as f:
        samples = pickle.load(f)

    load_info = {
        'cached_version': cached_version,
        'native_per_tf_labels': native_labels,
        'migrated': False,
        'migration_warnings': []
    }

    # Check if migration is needed (v7.x -> v8.0.0)
    if cached_version in COMPATIBLE_CACHE_VERSIONS and not native_labels:
        if migrate_labels:
            print(f"Migrating {len(samples)} samples from {cached_version} to v8.0.0 format...")
            migrated_count = 0
            for sample in samples:
                # v7.x samples have sample.labels as single ChannelLabels
                # v8.0.0 expects Dict[str, ChannelLabels]
                if isinstance(sample.labels, ChannelLabels):
                    # Wrap in dict with '5min' key (the base TF in v7)
                    sample.labels = {'5min': sample.labels}
                    migrated_count += 1
            load_info['migrated'] = True
            load_info['migration_warnings'].append(
                f"Migrated {migrated_count} samples to per-TF label format. "
                "Labels are only available for '5min' TF. "
                "Rebuild cache for native per-TF labels."
            )
            print(f"  Migrated {migrated_count} samples (labels wrapped for '5min' TF only)")
        else:
            load_info['migration_warnings'].append(
                f"Cache version {cached_version} uses legacy label format. "
                "Set migrate_labels=True or rebuild cache for native per-TF labels."
            )

    print(f"Loaded {len(samples)} samples from {cache_path}")
    print(f"  Cache version: {cached_version}")
    if native_labels:
        print(f"  Native per-TF labels: Yes")
    elif cached_version in COMPATIBLE_CACHE_VERSIONS:
        print(f"  Native per-TF labels: No (v7.x format)")

    return samples, load_info


def split_by_date(
    samples: List[ChannelSample],
    train_end: str = "2022-12-31",
    val_end: str = "2023-12-31"
) -> Tuple[List[ChannelSample], List[ChannelSample], List[ChannelSample]]:
    """
    Split samples into train/val/test by date.

    Default split:
    - Train: up to 2022-12-31
    - Val: 2023-01-01 to 2023-12-31
    - Test: 2024-01-01 onwards

    Args:
        samples: List of all samples
        train_end: End date for training set (inclusive)
        val_end: End date for validation set (inclusive)

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)

    train_samples = [s for s in samples if s.timestamp <= train_end_dt]
    val_samples = [s for s in samples if train_end_dt < s.timestamp <= val_end_dt]
    test_samples = [s for s in samples if s.timestamp > val_end_dt]

    print(f"Split: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")

    return train_samples, val_samples, test_samples


def collate_fn(batch: List[Tuple[Dict, Dict]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Collate function for DataLoader to batch samples together.

    Handles variable-length sequences by padding/stacking.

    Args:
        batch: List of (features_dict, labels_dict) tuples

    Returns:
        Tuple of (batched_features, batched_labels)
        - batched_features: Dict with stacked tensors of shape [batch_size, feature_dim]
        - batched_labels: Dict with stacked label tensors

    v9.0.0 batched_labels structure:
        Per-TF labels [batch, 11]:
            - duration, direction, next_channel, trigger_tf
        Per-label validity masks [batch, 11]:
            - duration_valid, direction_valid, next_channel_valid, trigger_tf_valid
        Aggregate [batch]:
            - duration_bars, permanent_break
    """
    # Separate features and labels
    features_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]

    # Stack features (all have same keys and shapes)
    batched_features = {}
    for key in features_list[0].keys():
        batched_features[key] = torch.stack([f[key] for f in features_list])

    # Stack labels (v9.0.0 format with separate validity masks)
    batched_labels = {
        # Per-timeframe labels - [batch, 11]
        'duration': torch.stack([l['duration'] for l in labels_list]),
        'direction': torch.stack([l['direction'] for l in labels_list]),
        'next_channel': torch.stack([l['next_channel'] for l in labels_list]),
        'trigger_tf': torch.stack([l['trigger_tf'] for l in labels_list]),

        # Per-label validity masks - [batch, 11]
        'duration_valid': torch.stack([l['duration_valid'] for l in labels_list]),
        'direction_valid': torch.stack([l['direction_valid'] for l in labels_list]),
        'next_channel_valid': torch.stack([l['next_channel_valid'] for l in labels_list]),
        'trigger_tf_valid': torch.stack([l['trigger_tf_valid'] for l in labels_list]),

        # Aggregate labels for reference/logging - [batch]
        'duration_bars': torch.stack([l['duration_bars'] for l in labels_list]),
        'permanent_break': torch.stack([l['permanent_break'] for l in labels_list]),
    }

    return batched_features, batched_labels


def create_dataloaders(
    train_samples: List[ChannelSample],
    val_samples: List[ChannelSample],
    test_samples: List[ChannelSample],
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    augment_train: bool = True,
    pin_memory: Optional[bool] = None,
    device: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create device-aware PyTorch DataLoaders for train/val/test sets.

    Automatically optimizes num_workers and pin_memory based on target device.
    Handles high-core-count systems (96+ cores) with intelligent scaling.

    Args:
        train_samples: Training samples
        val_samples: Validation samples
        test_samples: Test samples
        batch_size: Batch size
        num_workers: Number of workers (None = auto-detect based on device/CPU count)
        augment_train: Whether to augment training data
        pin_memory: Pin memory for faster GPU transfer (None = auto-detect based on device)
        device: Target device ('cuda', 'mps', 'cpu', or None for auto-detect)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    import os
    import torch

    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    # Auto-configure num_workers if not specified
    if num_workers is None:
        cpu_count = os.cpu_count() or 8

        if device == 'cuda':
            # Smart scaling for GPU training
            # Formula: min(cpu_count // 4, 24) for high-core systems
            # Caps at 24 workers (practical limit)
            if cpu_count >= 96:
                num_workers = 20  # 96-core systems: 20 workers
            elif cpu_count >= 64:
                num_workers = 16  # 64-core systems: 16 workers
            elif cpu_count >= 32:
                num_workers = 12  # 32-core systems: 12 workers
            elif cpu_count >= 16:
                num_workers = 6   # 16-core systems: 6 workers
            elif cpu_count >= 8:
                num_workers = 4   # 8-core systems: 4 workers
            else:
                num_workers = 2   # Low-core systems: 2 workers
        elif device == 'mps':
            # Apple Silicon: no multiprocessing benefit
            num_workers = 0
        else:  # CPU
            # CPU-only: minimal workers to avoid overhead
            num_workers = 0

    # Auto-configure pin_memory if not specified
    if pin_memory is None:
        pin_memory = (device == 'cuda')  # Only beneficial for CUDA

    # Print configuration for visibility
    print(f"DataLoader config: device={device}, num_workers={num_workers}, "
          f"pin_memory={pin_memory}, batch_size={batch_size}")

    # Create datasets
    train_dataset = ChannelDataset(train_samples, augment=augment_train)
    val_dataset = ChannelDataset(val_samples, augment=False)
    test_dataset = ChannelDataset(test_samples, augment=False)

    # Create dataloaders with device-aware settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),  # CRITICAL: Prevents file descriptor leaks
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)  # CRITICAL: Prevents file descriptor leaks
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)  # CRITICAL: Prevents file descriptor leaks
    )

    return train_loader, val_loader, test_loader


def prepare_dataset_from_scratch(
    data_dir: Path,
    cache_dir: Path,
    window: int = 50,
    step: int = 10,
    min_cycles: int = 1,
    max_scan: int = 500,
    return_threshold: int = 20,
    lookforward_bars: int = 200,
    train_end: str = "2022-12-31",
    val_end: str = "2023-12-31",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_history: bool = False,
    force_rebuild: bool = False,
    warn_on_mismatch: bool = True
) -> Tuple[List[ChannelSample], List[ChannelSample], List[ChannelSample]]:
    """
    Complete pipeline to prepare dataset from raw data.

    This function:
    1. Loads market data
    2. Scans for valid channels
    3. Caches samples to disk
    4. Splits into train/val/test

    Args:
        data_dir: Directory with CSV files
        cache_dir: Directory to store cache files
        window: Channel detection window
        step: Sliding window step
        min_cycles: Minimum cycles for valid channel
        max_scan: Maximum bars to scan forward for labels
        return_threshold: Bars outside channel to confirm break
        lookforward_bars: Bars to look forward for exit tracking
        train_end: End of training period
        val_end: End of validation period
        start_date: Optional start date for data
        end_date: Optional end date for data
        include_history: Include channel history features
        force_rebuild: Force rebuild cache even if exists
        warn_on_mismatch: Print warning if cache params don't match requested

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    cache_path = cache_dir / "channel_samples.pkl"

    # Check if cache exists and is valid
    cache_usable = is_cache_valid(cache_path) and not force_rebuild

    if cache_usable:
        # Validate parameters match cached values
        params_match, mismatches = validate_cache_params(
            cache_path,
            window=window,
            step=step,
            min_cycles=min_cycles,
            max_scan=max_scan,
            return_threshold=return_threshold,
            include_history=include_history,
            lookforward_bars=lookforward_bars
        )

        if not params_match:
            if warn_on_mismatch:
                print(f"\n⚠️  Cache parameter mismatch detected:")
                for mismatch in mismatches:
                    print(f"   - {mismatch}")
                print(f"   Cache will be rebuilt with new parameters.\n")
            cache_usable = False

    if cache_usable:
        print(f"Loading cached samples from {cache_path}")
        samples, load_info = load_cached_samples(cache_path, migrate_labels=True)
        # Show migration warnings if any
        for warning in load_info.get('migration_warnings', []):
            print(f"  Warning: {warning}")
    else:
        if cache_path.exists():
            if not is_cache_valid(cache_path):
                reason = "version mismatch"
            else:
                reason = "parameter mismatch"
            # Backup old cache before rebuilding
            backup_path = cache_path.with_name(f"{cache_path.stem}_old.pkl")
            old_meta_path = get_cache_metadata_path(cache_path)

            print(f"Cache {reason} - backing up old cache to {backup_path}")
            if backup_path.exists():
                print(f"  Removing existing backup: {backup_path}")
                backup_path.unlink()
            cache_path.rename(backup_path)

            if old_meta_path.exists():
                old_meta_backup = backup_path.with_suffix('.json')
                if old_meta_backup.exists():
                    old_meta_backup.unlink()
                old_meta_path.rename(old_meta_backup)

        print("Building dataset from scratch...")

        # Load data
        print("Loading market data...")
        tsla_df, spy_df, vix_df = load_market_data(
            data_dir,
            start_date=start_date,
            end_date=end_date
        )

        print(f"TSLA: {len(tsla_df)} bars ({tsla_df.index[0]} to {tsla_df.index[-1]})")
        print(f"SPY: {len(spy_df)} bars ({spy_df.index[0]} to {spy_df.index[-1]})")
        print(f"VIX: {len(vix_df)} bars ({vix_df.index[0]} to {vix_df.index[-1]})")

        # Scan for valid channels
        print(f"\nScanning for valid channels (window={window}, step={step})...")
        samples, min_warmup_bars = scan_valid_channels(
            tsla_df,
            spy_df,
            vix_df,
            window=window,
            step=step,
            min_cycles=min_cycles,
            max_scan=max_scan,
            return_threshold=return_threshold,
            include_history=include_history,
            lookforward_bars=lookforward_bars
        )

        print(f"\nFound {len(samples)} valid channel samples")

        # Collect which TFs have valid labels across all samples
        valid_tf_set = set()
        for sample in samples:
            if isinstance(sample.labels, dict):
                for tf, label in sample.labels.items():
                    if label is not None:
                        valid_tf_set.add(tf)

        # Cache to disk with ALL parameters (including warmup and v8 fields)
        metadata = {
            'window': window,
            'step': step,
            'min_cycles': min_cycles,
            'max_scan': max_scan,
            'return_threshold': return_threshold,
            'lookforward_bars': lookforward_bars,
            'include_history': include_history,
            'min_warmup_bars': min_warmup_bars,
            'num_samples': len(samples),
            'start_date': str(tsla_df.index[0]),
            'end_date': str(tsla_df.index[-1]),
            'created_at': str(datetime.now()),
            # v8.0.0 fields for native per-TF labels
            'native_per_tf_labels': True,
            'valid_tf_labels': sorted(list(valid_tf_set)),
            'label_generation_mode': 'per_tf_native',
        }
        cache_samples(samples, cache_path, metadata)

    # Split by date
    print("\nSplitting into train/val/test...")
    train_samples, val_samples, test_samples = split_by_date(
        samples,
        train_end=train_end,
        val_end=val_end
    )

    return train_samples, val_samples, test_samples


if __name__ == '__main__':
    """
    Example usage: Prepare dataset and create dataloaders.
    """
    # Setup paths
    data_dir = Path(__file__).parent.parent.parent / "data"
    cache_dir = Path(__file__).parent.parent.parent / "data" / "feature_cache"

    # Prepare dataset
    train_samples, val_samples, test_samples = prepare_dataset_from_scratch(
        data_dir=data_dir,
        cache_dir=cache_dir,
        window=50,
        step=25,  # Step=25 gives ~2x more samples than step=50
        min_cycles=1,
        train_end="2022-12-31",
        val_end="2023-12-31",
        include_history=False,  # Set True for full features (slower)
        force_rebuild=False
    )

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_samples,
        val_samples,
        test_samples,
        batch_size=32,
        num_workers=4,
        augment_train=True
    )

    # Test loading a batch
    print("\nTesting batch loading...")
    features, labels = next(iter(train_loader))

    print(f"\nBatch shape info:")
    print(f"Features keys: {list(features.keys())}")
    print(f"Sample feature shapes:")
    for k, v in list(features.items())[:3]:
        print(f"  {k}: {v.shape}")

    print(f"\nLabels:")
    print(f"  duration_bars: {labels['duration_bars'].shape}")
    print(f"  direction: {labels['direction'].shape}")
    print(f"  next_channel: {labels['next_channel'].shape}")
    print(f"  permanent_break: {labels['permanent_break'].shape}")

    print(f"\nDataloader sizes:")
    print(f"  Train: {len(train_loader)} batches ({len(train_samples)} samples)")
    print(f"  Val: {len(val_loader)} batches ({len(val_samples)} samples)")
    print(f"  Test: {len(test_loader)} batches ({len(test_samples)} samples)")
