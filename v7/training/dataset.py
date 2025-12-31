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
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import sys
from tqdm import tqdm
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.channel import detect_channel, Channel
from core.timeframe import resample_ohlc, TIMEFRAMES
from features.full_features import extract_full_features, features_to_tensor_dict, FullFeatures
from training.labels import generate_labels, ChannelLabels, labels_to_dict, labels_to_array


@dataclass
class ChannelSample:
    """
    A single training sample consisting of a valid channel + its features and labels.

    Attributes:
        timestamp: When this channel ends (last bar of detection window)
        channel_end_idx: Index in the full dataset where channel ends
        channel: The detected Channel object
        features: FullFeatures extracted at channel end
        labels: ChannelLabels from forward scan
    """
    timestamp: pd.Timestamp
    channel_end_idx: int
    channel: Channel
    features: FullFeatures
    labels: ChannelLabels


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

        # Convert labels to dict
        # NOTE: Currently we have one label per sample (5min channel)
        # We replicate it to all 11 timeframes for per-TF training
        # TODO Future: generate labels for each TF separately
        labels_dict = {
            'duration': torch.tensor([sample.labels.duration_bars] * 11, dtype=torch.float32),  # [11]
            'direction': torch.tensor([sample.labels.break_direction] * 11, dtype=torch.long),  # [11]
            'next_channel': torch.tensor([sample.labels.new_channel_direction] * 11, dtype=torch.long),  # [11]

            # Keep originals for reference/logging
            'duration_bars': torch.tensor(sample.labels.duration_bars, dtype=torch.float32),
            'break_trigger_tf': sample.labels.break_trigger_tf,  # String, not tensor
            'permanent_break': torch.tensor(int(sample.labels.permanent_break), dtype=torch.long),
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
    end_date: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load TSLA, SPY, and VIX data from CSV files.

    Args:
        data_dir: Directory containing CSV files
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        Tuple of (tsla_df, spy_df, vix_df)
    """
    # Load TSLA 1min data
    tsla_path = data_dir / "TSLA_1min.csv"
    tsla_df = pd.read_csv(tsla_path, parse_dates=['timestamp'])
    tsla_df.set_index('timestamp', inplace=True)
    tsla_df.columns = [c.lower() for c in tsla_df.columns]

    # Resample to 5min (base timeframe)
    tsla_df = resample_ohlc(tsla_df, '5min')

    # Load SPY 1min data
    spy_path = data_dir / "SPY_1min.csv"
    spy_df = pd.read_csv(spy_path, parse_dates=['timestamp'])
    spy_df.set_index('timestamp', inplace=True)
    spy_df.columns = [c.lower() for c in spy_df.columns]

    # Resample to 5min
    spy_df = resample_ohlc(spy_df, '5min')

    # Load VIX daily data
    vix_path = data_dir / "VIX_History.csv"
    vix_df = pd.read_csv(vix_path, parse_dates=['DATE'])
    vix_df.set_index('DATE', inplace=True)
    vix_df.columns = [c.lower() for c in vix_df.columns]

    # Filter by date range if provided
    if start_date:
        tsla_df = tsla_df[tsla_df.index >= start_date]
        spy_df = spy_df[spy_df.index >= start_date]
        vix_df = vix_df[vix_df.index >= start_date]

    if end_date:
        tsla_df = tsla_df[tsla_df.index <= end_date]
        spy_df = spy_df[spy_df.index <= end_date]
        vix_df = vix_df[vix_df.index <= end_date]

    return tsla_df, spy_df, vix_df


def scan_valid_channels(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    window: int = 50,
    step: int = 10,
    min_cycles: int = 1,
    max_scan: int = 500,
    return_threshold: int = 20,
    include_history: bool = False,
    lookforward_bars: int = 200,
    progress: bool = True
) -> List[ChannelSample]:
    """
    Scan through historical data to find all valid channels and generate samples.

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

    # Need enough data for channel window + forward scan
    min_required = window + max_scan

    # Align SPY and VIX with TSLA timestamps
    # For VIX (daily), forward-fill to match 5min timestamps
    vix_aligned = vix_df.reindex(tsla_df.index, method='ffill')

    # Scan through data with sliding window
    start_idx = window
    end_idx = len(tsla_df) - max_scan

    indices = range(start_idx, end_idx, step)
    if progress:
        indices = tqdm(indices, desc="Scanning channels")

    for i in indices:
        # Get data window
        tsla_window = tsla_df.iloc[:i]
        spy_window = spy_df.iloc[:i]
        vix_window = vix_aligned.iloc[:i]

        if len(tsla_window) < window or len(spy_window) < window:
            continue

        # Detect channel on TSLA
        channel = detect_channel(tsla_window, window=window, min_cycles=min_cycles)

        if not channel.valid:
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
            # Skip if feature extraction fails
            if progress:
                tqdm.write(f"Feature extraction failed at {i}: {e}")
            continue

        # Generate labels by scanning forward
        try:
            labels = generate_labels(
                df=tsla_df,
                channel=channel,
                channel_end_idx=i - 1,
                current_tf='5min',
                window=window,
                max_scan=max_scan,
                return_threshold=return_threshold
            )
        except Exception as e:
            # Skip if label generation fails
            if progress:
                tqdm.write(f"Label generation failed at {i}: {e}")
            continue

        # Create sample
        sample = ChannelSample(
            timestamp=tsla_df.index[i - 1],
            channel_end_idx=i - 1,
            channel=channel,
            features=features,
            labels=labels
        )

        samples.append(sample)

    return samples


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

    # Save metadata as JSON
    if metadata:
        meta_path = cache_path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            # Convert datetime objects to strings for JSON
            meta_serializable = {}
            for k, v in metadata.items():
                if isinstance(v, (datetime, pd.Timestamp)):
                    meta_serializable[k] = str(v)
                else:
                    meta_serializable[k] = v
            json.dump(meta_serializable, f, indent=2)

    print(f"Cached {len(samples)} samples to {cache_path}")


def load_cached_samples(cache_path: Path) -> List[ChannelSample]:
    """
    Load cached samples from disk.

    Args:
        cache_path: Path to cache file (.pkl)

    Returns:
        List of ChannelSample objects
    """
    with open(cache_path, 'rb') as f:
        samples = pickle.load(f)

    print(f"Loaded {len(samples)} samples from {cache_path}")
    return samples


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
    """
    # Separate features and labels
    features_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]

    # Stack features (all have same keys and shapes)
    batched_features = {}
    for key in features_list[0].keys():
        batched_features[key] = torch.stack([f[key] for f in features_list])

    # Stack labels (handle both per-TF and original formats)
    batched_labels = {
        # Per-timeframe labels for CombinedLoss - [batch, 11]
        'duration': torch.stack([l['duration'] for l in labels_list]),
        'direction': torch.stack([l['direction'] for l in labels_list]),
        'next_channel': torch.stack([l['next_channel'] for l in labels_list]),

        # Original labels for reference/logging
        'duration_bars': torch.stack([l['duration_bars'] for l in labels_list]),
        'break_trigger_tf': [l['break_trigger_tf'] for l in labels_list],  # Keep as list
        'permanent_break': torch.stack([l['permanent_break'] for l in labels_list]),
    }

    return batched_features, batched_labels


def create_dataloaders(
    train_samples: List[ChannelSample],
    val_samples: List[ChannelSample],
    test_samples: List[ChannelSample],
    batch_size: int = 32,
    num_workers: int = 4,
    augment_train: bool = True,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test sets.

    Args:
        train_samples: Training samples
        val_samples: Validation samples
        test_samples: Test samples
        batch_size: Batch size
        num_workers: Number of workers for data loading
        augment_train: Whether to augment training data
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = ChannelDataset(train_samples, augment=augment_train)
    val_dataset = ChannelDataset(val_samples, augment=False)
    test_dataset = ChannelDataset(test_samples, augment=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch for stable training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


def prepare_dataset_from_scratch(
    data_dir: Path,
    cache_dir: Path,
    window: int = 50,
    step: int = 10,
    min_cycles: int = 1,
    train_end: str = "2022-12-31",
    val_end: str = "2023-12-31",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_history: bool = False,
    force_rebuild: bool = False
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
        train_end: End of training period
        val_end: End of validation period
        start_date: Optional start date for data
        end_date: Optional end date for data
        include_history: Include channel history features
        force_rebuild: Force rebuild cache even if exists

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    cache_path = cache_dir / "channel_samples.pkl"

    # Check if cache exists
    if cache_path.exists() and not force_rebuild:
        print(f"Loading cached samples from {cache_path}")
        samples = load_cached_samples(cache_path)
    else:
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
        samples = scan_valid_channels(
            tsla_df,
            spy_df,
            vix_df,
            window=window,
            step=step,
            min_cycles=min_cycles,
            include_history=include_history
        )

        print(f"\nFound {len(samples)} valid channel samples")

        # Cache to disk
        metadata = {
            'window': window,
            'step': step,
            'min_cycles': min_cycles,
            'num_samples': len(samples),
            'start_date': str(tsla_df.index[0]),
            'end_date': str(tsla_df.index[-1]),
            'include_history': include_history,
            'created_at': str(datetime.now()),
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
    print(f"  break_direction: {labels['break_direction'].shape}")
    print(f"  new_channel_direction: {labels['new_channel_direction'].shape}")
    print(f"  permanent_break: {labels['permanent_break'].shape}")

    print(f"\nDataloader sizes:")
    print(f"  Train: {len(train_loader)} batches ({len(train_samples)} samples)")
    print(f"  Val: {len(val_loader)} batches ({len(val_samples)} samples)")
    print(f"  Test: {len(test_loader)} batches ({len(test_samples)} samples)")
